import asyncio
import audioop
import logging
import math
import os
import socket
import time
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv

from tts_engine import TTSEngine


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice_bot")

RTP_LISTEN_IP = os.getenv("RTP_LISTEN_IP", "0.0.0.0")
RTP_LISTEN_PORT = int(os.getenv("RTP_LISTEN_PORT", "10000"))

ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")
ASR_MODEL = os.getenv("ASR_MODEL", "large-v3-turbo")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "ru")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_URL") or "http://llm-server:8085/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-local-key")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты вежливый виртуальный оператор. Отвечай кратко и по делу.")

SAMPLE_RATE_TEL = int(os.getenv("SAMPLE_RATE_TEL", "8000"))
CODEC_ENV = (os.getenv("CODEC", "pcmu") or "pcmu").strip().lower()

FRAME_MS = int(os.getenv("FRAME_MS", "20"))
MAX_UTTERANCE_SEC = float(os.getenv("MAX_UTTERANCE_SEC", "12"))
SILENCE_SEC = float(os.getenv("SILENCE_SEC", "0.9"))
MIN_SPEECH_SEC = float(os.getenv("MIN_SPEECH_SEC", "0.35"))
RMS_THRESHOLD = int(os.getenv("RMS_THRESHOLD", "220"))

OUT_GAIN = float(os.getenv("OUT_GAIN", "1.0"))

RECORD_CALLS = (os.getenv("RECORD_CALLS", "1") == "1")
RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "/recordings").rstrip("/")

TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda")
TTS_REF_AUDIO = os.getenv("TTS_REF_AUDIO", "/app/ref_audio.wav")

CHUNK_SAMPLES = int(SAMPLE_RATE_TEL * FRAME_MS / 1000)
if CHUNK_SAMPLES <= 0:
    CHUNK_SAMPLES = 160


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return audio
    up = int(sr_to)
    down = int(sr_from)
    g = math.gcd(up, down)
    up //= g
    down //= g
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def _normalize_codec_name(codec: str) -> str:
    c = (codec or "").strip().lower()
    if c in ("pcmu", "ulaw", "mu", "mulaw", "g711u"):
        return "ulaw"
    if c in ("pcma", "alaw", "a", "g711a"):
        return "alaw"
    return "ulaw"


def _codec_from_payload_type(pt: int) -> str | None:
    if pt == 0:
        return "ulaw"
    if pt == 8:
        return "alaw"
    return None


def _payload_type_from_codec(codec: str) -> int:
    c = _normalize_codec_name(codec)
    return 8 if c == "alaw" else 0


def decode_payload(payload: bytes, codec: str) -> np.ndarray:
    c = _normalize_codec_name(codec)
    if c == "alaw":
        pcm = audioop.alaw2lin(payload, 2)
    else:
        pcm = audioop.ulaw2lin(payload, 2)
    audio_i16 = np.frombuffer(pcm, dtype=np.int16)
    return audio_i16.astype(np.float32, copy=False) / 32768.0


def encode_payload(audio_f32: np.ndarray, codec: str) -> bytes:
    c = _normalize_codec_name(codec)
    a = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (a * 32767.0).astype(np.int16, copy=False)
    pcm = audio_i16.tobytes()
    if c == "alaw":
        return audioop.lin2alaw(pcm, 2)
    return audioop.lin2ulaw(pcm, 2)


def parse_rtp(data: bytes):
    if len(data) < 12:
        return None
    vpxcc = data[0]
    version = vpxcc >> 6
    if version != 2:
        return None
    padding = (vpxcc & 0x20) != 0
    extension = (vpxcc & 0x10) != 0
    csrc_count = vpxcc & 0x0F

    mpt = data[1]
    payload_type = mpt & 0x7F

    seq = int.from_bytes(data[2:4], "big")
    ts = int.from_bytes(data[4:8], "big")
    ssrc = int.from_bytes(data[8:12], "big")

    offset = 12 + 4 * csrc_count
    if len(data) < offset:
        return None

    if extension:
        if len(data) < offset + 4:
            return None
        ext_len_words = int.from_bytes(data[offset + 2 : offset + 4], "big")
        offset = offset + 4 + 4 * ext_len_words
        if len(data) < offset:
            return None

    payload = data[offset:]
    if padding:
        if not payload:
            return None
        pad_len = payload[-1]
        if pad_len <= 0 or pad_len > len(payload):
            return None
        payload = payload[:-pad_len]

    return payload_type, seq, ts, ssrc, payload


def _f32_to_i16_bytes(audio_f32: np.ndarray) -> bytes:
    a = np.clip(audio_f32, -1.0, 1.0)
    x = (a * 32767.0).astype(np.int16, copy=False)
    return x.tobytes()


def _open_wav(path: Path, sr: int, channels: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    wf = wave.open(str(path), "wb")
    wf.setnchannels(int(channels))
    wf.setsampwidth(2)
    wf.setframerate(int(sr))
    return wf


class CallRecorder:
    def __init__(self, base_dir: str, addr: tuple[str, int]):
        self.enabled = False
        self.dir = None
        self.rx = None
        self.tx = None
        self.tx_sent = None
        self.tr = None
        self.meta = None

        if not RECORD_CALLS:
            return
        if not base_dir:
            return

        try:
            date_dir = time.strftime("%Y-%m-%d", time.localtime())
            ts = int(time.time() * 1000)
            ip = addr[0].replace(".", "-").replace(":", "-")
            call_dir = Path(base_dir) / date_dir / f"{ts}_{ip}_{addr[1]}"
            call_dir.mkdir(parents=True, exist_ok=True)
            self.dir = call_dir

            self.rx = _open_wav(call_dir / "rx.wav", SAMPLE_RATE_TEL, 1)
            self.tx = _open_wav(call_dir / "tx.wav", SAMPLE_RATE_TEL, 1)
            self.tx_sent = _open_wav(call_dir / "tx_sent.wav", SAMPLE_RATE_TEL, 1)

            self.tr = open(call_dir / "transcript.txt", "a", encoding="utf-8", buffering=1)
            self.meta = open(call_dir / "meta.txt", "a", encoding="utf-8", buffering=1)

            self.meta.write(f"peer={addr[0]}:{addr[1]}\n")
            self.meta.write(f"start={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

            self.enabled = True
        except Exception:
            self.enabled = False

    def write_rx(self, audio_f32: np.ndarray):
        if not self.enabled or self.rx is None:
            return
        try:
            self.rx.writeframes(_f32_to_i16_bytes(audio_f32))
        except Exception:
            pass

    def write_tx(self, audio_f32: np.ndarray):
        if not self.enabled or self.tx is None:
            return
        try:
            self.tx.writeframes(_f32_to_i16_bytes(audio_f32))
        except Exception:
            pass

    def write_tx_sent(self, audio_f32: np.ndarray):
        if not self.enabled or self.tx_sent is None:
            return
        try:
            self.tx_sent.writeframes(_f32_to_i16_bytes(audio_f32))
        except Exception:
            pass

    def write_line(self, line: str):
        if not self.enabled or self.tr is None:
            return
        try:
            self.tr.write(line + "\n")
        except Exception:
            pass

    def write_meta(self, line: str):
        if not self.enabled or self.meta is None:
            return
        try:
            self.meta.write(line + "\n")
        except Exception:
            pass

    def close(self):
        try:
            if self.meta is not None:
                self.meta.write(f"end={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        except Exception:
            pass

        for f in (self.rx, self.tx, self.tx_sent):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

        for f in (self.tr, self.meta):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass


class CallHandler:
    def __init__(self, addr, sock, stt_model: WhisperModel, tts_engine: TTSEngine, openai_client: OpenAI):
        self.addr = addr
        self.sock = sock
        self.stt_model = stt_model
        self.tts_engine = tts_engine
        self.openai_client = openai_client

        self.buffer = bytearray()
        self.last_voice_time = time.time()
        self.speech_started = False
        self.voice_samples = 0
        self.start_time = time.time()

        self.out_seq = 0
        self.out_ts = 0
        self.ssrc = int.from_bytes(os.urandom(4), "big")

        self.codec = _normalize_codec_name(CODEC_ENV)
        self.out_pt = _payload_type_from_codec(self.codec)

        self.frame_samples = None
        self.last_in_ts = None
        self.logged_peer = False

        self.last_packet_time = time.time()
        self.busy = False

        self.rec = CallRecorder(RECORDINGS_DIR, (addr[0], addr[1]))

    def close(self):
        try:
            self.rec.close()
        except Exception:
            pass

    def _maybe_log_peer(self, in_pt: int, in_len: int):
        if self.logged_peer:
            return
        self.logged_peer = True
        logger.info(
            "RTP peer %s:%d in_pt=%d in_len=%d codec=%s out_pt=%d",
            self.addr[0],
            self.addr[1],
            int(in_pt),
            int(in_len),
            self.codec,
            int(self.out_pt),
        )
        self.rec.write_meta(f"in_pt={int(in_pt)} in_len={int(in_len)} codec={self.codec} out_pt={int(self.out_pt)}")

    def _set_stream_info(self, in_pt: int, in_ts: int, in_len: int):
        self.last_in_ts = int(in_ts)
        c = _codec_from_payload_type(int(in_pt))
        if c is not None:
            self.codec = c
            self.out_pt = _payload_type_from_codec(self.codec)
        if in_len > 0:
            self.frame_samples = int(in_len)

    def feed(self, payload_type, ts, payload):
        self.last_packet_time = time.time()
        self._maybe_log_peer(int(payload_type), len(payload))
        self._set_stream_info(int(payload_type), int(ts), len(payload))

        c = _codec_from_payload_type(int(payload_type))
        if c is None:
            return

        audio = decode_payload(payload, self.codec)
        self.rec.write_rx(audio)

        rms = float(math.sqrt(np.mean(audio * audio))) * 32768.0

        now = time.time()
        if rms > RMS_THRESHOLD:
            self.speech_started = True
            self.last_voice_time = now
            self.voice_samples += int(audio.shape[0])
            self.buffer.extend(audio.tobytes())
        else:
            if self.speech_started:
                self.buffer.extend(audio.tobytes())

        if not self.busy and self.speech_started:
            speech_sec = float(self.voice_samples) / float(SAMPLE_RATE_TEL)
            silence_sec = now - self.last_voice_time
            total_sec = now - self.start_time

            if total_sec >= MAX_UTTERANCE_SEC:
                asyncio.create_task(self.flush())
            elif silence_sec >= SILENCE_SEC and speech_sec >= MIN_SPEECH_SEC:
                asyncio.create_task(self.flush())

    async def flush(self):
        if self.busy:
            return
        self.busy = True
        try:
            raw = bytes(self.buffer)
            self.buffer = bytearray()
            self.speech_started = False
            self.voice_samples = 0
            self.start_time = time.time()

            min_samples = int(SAMPLE_RATE_TEL * MIN_SPEECH_SEC)
            if len(raw) < 4 * min_samples:
                return

            audio_tel = np.frombuffer(raw, dtype=np.float32)

            audio_16k = resample(audio_tel, SAMPLE_RATE_TEL, 16000)
            audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32, copy=False)

            segments, _ = self.stt_model.transcribe(audio_16k, language=ASR_LANGUAGE, vad_filter=True)
            text = " ".join([s.text.strip() for s in segments]).strip()
            if not text:
                return

            logger.info("User: %s", text)
            self.rec.write_line(f"USER: {text}")

            reply = await self.ask_llm(text)
            if not reply:
                return

            logger.info("Bot: %s", reply)
            self.rec.write_line(f"BOT: {reply}")

            audio_tts, sr_tts = await self.tts_engine.generate(reply)
            if audio_tts is None or sr_tts is None:
                return

            audio_out = resample(audio_tts, int(sr_tts), SAMPLE_RATE_TEL)
            if OUT_GAIN != 1.0:
                audio_out = (audio_out * float(OUT_GAIN)).astype(np.float32, copy=False)

            self.rec.write_tx(audio_out)

            try:
                payload_all = encode_payload(audio_out, self.codec)
                decoded_all = decode_payload(payload_all, self.codec)
                self.rec.write_tx_sent(decoded_all)
            except Exception:
                pass

            await self.stream_audio_back(audio_out)
        finally:
            self.busy = False

    async def ask_llm(self, user_text):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            resp = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=256,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error("LLM error: %s", e)
            return ""

    async def stream_audio_back(self, audio_tel_f32: np.ndarray):
        if audio_tel_f32.size == 0:
            return

        frame_samples = int(self.frame_samples) if self.frame_samples else int(CHUNK_SAMPLES)
        if frame_samples <= 0:
            frame_samples = int(CHUNK_SAMPLES)

        frame_sec = float(frame_samples) / float(SAMPLE_RATE_TEL)

        pcm_payload = encode_payload(audio_tel_f32, self.codec)

        pad_byte = b"\xd5" if self.codec == "alaw" else b"\xff"
        rem = len(pcm_payload) % frame_samples
        if rem != 0:
            pcm_payload += pad_byte * (frame_samples - rem)

        if self.last_in_ts is not None:
            self.out_ts = (int(self.last_in_ts) + int(frame_samples)) & 0xFFFFFFFF

        for i in range(0, len(pcm_payload), frame_samples):
            payload = pcm_payload[i : i + frame_samples]
            header = bytearray(12)
            header[0] = 0x80
            header[1] = int(self.out_pt) & 0x7F
            header[2:4] = int(self.out_seq).to_bytes(2, "big")
            header[4:8] = int(self.out_ts).to_bytes(4, "big")
            header[8:12] = int(self.ssrc).to_bytes(4, "big")
            packet = bytes(header) + payload
            self.sock.sendto(packet, self.addr)
            self.out_seq = (int(self.out_seq) + 1) & 0xFFFF
            self.out_ts = (int(self.out_ts) + int(frame_samples)) & 0xFFFFFFFF
            await asyncio.sleep(frame_sec)


async def main():
    logger.info("Starting voice bot on %s:%d sr=%d codec=%s", RTP_LISTEN_IP, RTP_LISTEN_PORT, SAMPLE_RATE_TEL, CODEC_ENV)

    if ASR_MODEL == "large-v3-turbo":
        snapshots = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "voice"
            / "hf"
            / "hub"
            / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
            / "snapshots"
        )
        if snapshots.is_dir():
            dirs = sorted([p for p in snapshots.iterdir() if p.is_dir()])
            if dirs:
                model_path = str(dirs[-1])
            else:
                model_path = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
        else:
            model_path = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    else:
        model_path = ASR_MODEL

    stt_model = WhisperModel(
        model_path,
        device=ASR_DEVICE,
        compute_type="float16" if ASR_DEVICE == "cuda" else "int8",
    )
    tts_engine = TTSEngine(ref_audio_path=TTS_REF_AUDIO, device=TTS_DEVICE)
    openai_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((RTP_LISTEN_IP, RTP_LISTEN_PORT))
    sock.setblocking(False)

    calls = {}

    async def cleanup_loop():
        while True:
            now = time.time()
            dead = []
            for addr, h in calls.items():
                if now - h.last_packet_time > 30:
                    dead.append(addr)
            for addr in dead:
                h = calls.pop(addr, None)
                if h is not None:
                    try:
                        h.close()
                    except Exception:
                        pass
            await asyncio.sleep(5)

    asyncio.create_task(cleanup_loop())

    loop = asyncio.get_running_loop()
    while True:
        data, addr = await loop.sock_recvfrom(sock, 4096)
        parsed = parse_rtp(data)
        if not parsed:
            continue
        payload_type, seq, ts, ssrc, payload = parsed
        if addr not in calls:
            calls[addr] = CallHandler(addr, sock, stt_model, tts_engine, openai_client)
        calls[addr].feed(payload_type, ts, payload)


if __name__ == "__main__":
    asyncio.run(main())
