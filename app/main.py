import asyncio
import audioop
import logging
import math
import os
import socket
import time
import numpy as np
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv
from tts_engine import TTSEngine
from pathlib import Path


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice_bot")


RTP_LISTEN_IP = os.getenv("RTP_LISTEN_IP", "0.0.0.0")
RTP_LISTEN_PORT = int(os.getenv("RTP_LISTEN_PORT", "10000"))

ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")
ASR_MODEL = os.getenv("ASR_MODEL", "large-v3-turbo")

if ASR_MODEL == "large-v3-turbo":
    _snapshots = (Path(__file__).resolve().parents[1] / "models" / "voice" / "hf" / "hub" / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo" / "snapshots")
    if _snapshots.is_dir():
        _dirs = sorted([p for p in _snapshots.iterdir() if p.is_dir()])
        if _dirs:
            ASR_MODEL = str(_dirs[-1])
        else:
            ASR_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    else:
        ASR_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "ru")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://llm-server:8085/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-local-key")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты вежливый виртуальный оператор. Отвечай кратко и по делу.")

SAMPLE_RATE_TEL = int(os.getenv("SAMPLE_RATE_TEL", "8000"))
CODEC = os.getenv("CODEC", "pcmu").lower()
FRAME_MS = int(os.getenv("FRAME_MS", "20"))
MAX_UTTERANCE_SEC = float(os.getenv("MAX_UTTERANCE_SEC", "12"))
SILENCE_SEC = float(os.getenv("SILENCE_SEC", "0.9"))
MIN_SPEECH_SEC = float(os.getenv("MIN_SPEECH_SEC", "0.35"))
RMS_THRESHOLD = int(os.getenv("RMS_THRESHOLD", "220"))

TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda")
TTS_REF_AUDIO = os.getenv("TTS_REF_AUDIO", "/app/ref_audio.wav")

CHUNK_SAMPLES = int(SAMPLE_RATE_TEL * FRAME_MS / 1000)
if CHUNK_SAMPLES <= 0:
    CHUNK_SAMPLES = 160

if CODEC not in ("pcmu", "ulaw", "alaw"):
    CODEC = "pcmu"


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return audio
    up = int(sr_to)
    down = int(sr_from)
    g = math.gcd(up, down)
    up //= g
    down //= g
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def decode_payload(payload: bytes) -> np.ndarray:
    if CODEC in ("pcmu", "ulaw"):
        pcm = audioop.ulaw2lin(payload, 2)
    else:
        pcm = audioop.alaw2lin(payload, 2)
    audio_i16 = np.frombuffer(pcm, dtype=np.int16)
    return audio_i16.astype(np.float32) / 32768.0


def encode_payload(audio_f32: np.ndarray) -> bytes:
    a = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (a * 32767.0).astype(np.int16)
    pcm = audio_i16.tobytes()
    if CODEC in ("pcmu", "ulaw"):
        return audioop.lin2ulaw(pcm, 2)
    return audioop.lin2alaw(pcm, 2)


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
        ext_len_words = int.from_bytes(data[offset + 2:offset + 4], "big")
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
        self.voice_frames = 0
        self.start_time = time.time()

        self.out_seq = 0
        self.out_ts = 0
        self.ssrc = int.from_bytes(os.urandom(4), "big")
        self.payload_type = 0

        self.first_inbound_ts = None
        self.last_packet_time = time.time()
        self.busy = False

    def feed(self, payload_type, ts, payload):
        self.last_packet_time = time.time()

        if self.first_inbound_ts is None:
            self.payload_type = int(payload_type)
            self.first_inbound_ts = int(ts)
            self.out_ts = int(ts)

        audio = decode_payload(payload)
        rms = float(math.sqrt(np.mean(audio * audio))) * 32768.0

        now = time.time()
        if rms > RMS_THRESHOLD:
            self.speech_started = True
            self.last_voice_time = now
            self.voice_frames += 1
            self.buffer.extend(audio.tobytes())
        else:
            if self.speech_started:
                self.buffer.extend(audio.tobytes())

        if not self.busy and self.speech_started:
            speech_sec = self.voice_frames * (FRAME_MS / 1000.0)
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
            self.voice_frames = 0
            self.start_time = time.time()

            min_samples = int(SAMPLE_RATE_TEL * MIN_SPEECH_SEC)
            if len(raw) < 4 * min_samples:
                return

            audio_tel = np.frombuffer(raw, dtype=np.float32)

            audio_16k = resample(audio_tel, SAMPLE_RATE_TEL, 16000)
            audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32, copy=False)

            segments, _ = self.stt_model.transcribe(
                audio_16k,
                language=ASR_LANGUAGE,
                vad_filter=True
            )
            text = " ".join([s.text.strip() for s in segments]).strip()
            if not text:
                return

            logger.info("User: %s", text)

            reply = await self.ask_llm(text)
            if not reply:
                return

            logger.info("Bot: %s", reply)

            audio_tts, sr_tts = await self.tts_engine.generate(reply)
            if audio_tts is None or sr_tts is None:
                return

            audio_out = resample(audio_tts, sr_tts, SAMPLE_RATE_TEL)

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
                max_tokens=256
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error("LLM error: %s", e)
            return ""

    async def stream_audio_back(self, audio_tel_f32: np.ndarray):
        if audio_tel_f32.size == 0:
            return

        pcm_payload = encode_payload(audio_tel_f32)

        frame_bytes = CHUNK_SAMPLES
        pad_byte = b"\xff" if CODEC in ("pcmu", "ulaw") else b"\xd5"

        rem = len(pcm_payload) % frame_bytes
        if rem != 0:
            pcm_payload += pad_byte * (frame_bytes - rem)

        for i in range(0, len(pcm_payload), frame_bytes):
            payload = pcm_payload[i:i + frame_bytes]
            header = bytearray(12)
            header[0] = 0x80
            header[1] = self.payload_type & 0x7F
            header[2:4] = self.out_seq.to_bytes(2, "big")
            header[4:8] = self.out_ts.to_bytes(4, "big")
            header[8:12] = self.ssrc.to_bytes(4, "big")
            packet = bytes(header) + payload
            self.sock.sendto(packet, self.addr)
            self.out_seq = (self.out_seq + 1) & 0xFFFF
            self.out_ts = (self.out_ts + CHUNK_SAMPLES) & 0xFFFFFFFF
            await asyncio.sleep(FRAME_MS / 1000.0)


async def main():
    logger.info("Starting voice bot on %s:%d codec=%s sr=%d", RTP_LISTEN_IP, RTP_LISTEN_PORT, CODEC, SAMPLE_RATE_TEL)

    stt_model = WhisperModel(
        ASR_MODEL,
        device=ASR_DEVICE,
        compute_type="float16" if ASR_DEVICE == "cuda" else "int8"
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
                calls.pop(addr, None)
            await asyncio.sleep(5)

    asyncio.create_task(cleanup_loop())

    loop = asyncio.get_running_loop()
    while True:
        data, addr = await loop.sock_recvfrom(sock, 2048)
        parsed = parse_rtp(data)
        if not parsed:
            continue
        payload_type, seq, ts, ssrc, payload = parsed
        if addr not in calls:
            calls[addr] = CallHandler(addr, sock, stt_model, tts_engine, openai_client)
        calls[addr].feed(payload_type, ts, payload)


if __name__ == "__main__":
    asyncio.run(main())
