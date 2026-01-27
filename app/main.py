# -*- coding: utf-8 -*-
import asyncio
import audioop
import numpy as np
import os
import sys
import struct
import time
import scipy.signal
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__).info
error_log = logging.getLogger(__name__).error

try:
    from faster_whisper import WhisperModel
    from openai import OpenAI
    from tts_engine import F5TTSWrapper
    import prompts
except ImportError as e:
    error_log(f"CRITICAL ERROR importing libraries: {e}")
    sys.exit(1)

RTP_IP = "0.0.0.0"
RTP_PORT = 10000
SAMPLE_RATE_TELEPHONY = 8000
SAMPLE_RATE_WHISPER = 16000
CHUNK_SIZE_20MS = 160

SPEECH_RMS_THRESHOLD = int(os.getenv("SPEECH_RMS_THRESHOLD", "260"))
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "65"))
PREROLL_FRAMES = int(os.getenv("PREROLL_FRAMES", "20"))
MIN_VOICE_BYTES = int(os.getenv("MIN_VOICE_BYTES", "8000"))
FORCE_RTP_PT = os.getenv("FORCE_RTP_PT", "").strip()

stt_model = None
llm_client = None
tts_engine = None

def load_models():
    global stt_model, llm_client, tts_engine
    stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    llm_client = OpenAI(base_url=os.getenv("LLM_API_URL"), api_key="sk-local-key")
    tts_engine = F5TTSWrapper(stt_model=stt_model)
    log("All models loaded.")

def rtp_parse(data: bytes):
    if not data or len(data) < 12:
        return None, None
    b0 = data[0]
    b1 = data[1]
    version = b0 >> 6
    if version != 2:
        return None, None
    x = (b0 >> 4) & 0x01
    cc = b0 & 0x0F
    pt = b1 & 0x7F
    off = 12 + (cc * 4)
    if off > len(data):
        return None, None
    if x:
        if off + 4 > len(data):
            return None, None
        ext_len_words = struct.unpack("!H", data[off + 2: off + 4])[0]
        off = off + 4 + (ext_len_words * 4)
        if off > len(data):
            return None, None
    return data[off:], pt

def pcm_decode(payload: bytes, pt: int):
    if pt == 8:
        return audioop.alaw2lin(payload, 2)
    return audioop.ulaw2lin(payload, 2)

def pcm_encode(pcm_int16_bytes: bytes, pt: int):
    if pt == 8:
        return audioop.lin2alaw(pcm_int16_bytes, 2)
    return audioop.lin2ulaw(pcm_int16_bytes, 2)

class CallHandler:
    def __init__(self, transport, addr):
        self.transport = transport
        self.client_addr = addr
        self.history = []
        self.seq_num = 0
        self.timestamp = 0
        self.ssrc = 123456

        self.is_speaking = False
        self.greeting_sent = False

        self.in_speech = False
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.preroll = deque(maxlen=PREROLL_FRAMES)

        self.rtp_pt = 0

    def _select_pt(self, incoming_pt: int):
        if FORCE_RTP_PT.isdigit():
            v = int(FORCE_RTP_PT)
            if v in (0, 8):
                return v
        if incoming_pt in (0, 8):
            return incoming_pt
        return 0

    async def send_greeting(self):
        if self.greeting_sent:
            return
        self.greeting_sent = True
        try:
            sr, audio = tts_engine.generate("Алло, да, я вас слушаю.")
            await self.stream_audio_back(audio, sr)
        except Exception as e:
            error_log(f"Greeting error: {e}")

    def process_incoming_audio(self, data):
        payload, pt = rtp_parse(data)
        if not payload:
            return

        if pt is not None:
            self.rtp_pt = self._select_pt(int(pt))

        try:
            pcm_data = pcm_decode(payload, self.rtp_pt)
        except Exception:
            return

        rms = audioop.rms(pcm_data, 2)
        self.preroll.append(pcm_data)

        if self.in_speech:
            self.audio_buffer.extend(pcm_data)
            if rms > SPEECH_RMS_THRESHOLD:
                self.silence_frames = 0
            else:
                self.silence_frames += 1
        else:
            if rms > SPEECH_RMS_THRESHOLD:
                self.in_speech = True
                self.silence_frames = 0
                for fr in self.preroll:
                    self.audio_buffer.extend(fr)
                self.preroll.clear()
                self.audio_buffer.extend(pcm_data)

        if self.in_speech and self.silence_frames > END_SILENCE_FRAMES and len(self.audio_buffer) >= MIN_VOICE_BYTES:
            audio_to_process = bytes(self.audio_buffer)
            self.audio_buffer = bytearray()
            self.in_speech = False
            self.silence_frames = 0
            asyncio.create_task(self.handle_turn(audio_to_process))

        if self.in_speech and self.silence_frames > (END_SILENCE_FRAMES * 4):
            self.audio_buffer = bytearray()
            self.in_speech = False
            self.silence_frames = 0

    async def handle_turn(self, audio_bytes):
        if not audio_bytes or self.is_speaking:
            return
        self.is_speaking = True

        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_np) < 800:
                return

            audio_16k = scipy.signal.resample_poly(audio_np, SAMPLE_RATE_WHISPER, SAMPLE_RATE_TELEPHONY)

            segments, _ = stt_model.transcribe(
                audio_16k,
                language="ru",
                beam_size=3,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 250}
            )
            user_text = " ".join([s.text for s in segments]).strip()

            if len(user_text) < 2:
                return

            log(f"User: {user_text}")
            self.history.append({"role": "user", "content": user_text})

            messages = prompts.create_messages(self.history)
            completion = await asyncio.to_thread(
                llm_client.chat.completions.create,
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=messages,
                temperature=0.3,
                max_tokens=220
            )
            bot_text = (completion.choices[0].message.content or "").strip()
            if not bot_text:
                return

            log(f"Bot: {bot_text}")
            self.history.append({"role": "assistant", "content": bot_text})

            sr_out, audio_out = await asyncio.to_thread(tts_engine.generate, bot_text)
            await self.stream_audio_back(audio_out, sr_out)

        except Exception as e:
            error_log(f"Error: {e}")
        finally:
            self.is_speaking = False

    async def stream_audio_back(self, audio_np, sr_in):
        if audio_np is None or len(audio_np) == 0:
            error_log("TTS returned empty audio")
            return

        duration = len(audio_np) / float(sr_in)
        if duration < 0.08:
            return

        max_val = float(np.max(np.abs(audio_np)))
        if max_val > 0.05:
            audio_np = audio_np / max_val * 0.90

        gcd = np.gcd(sr_in, SAMPLE_RATE_TELEPHONY)
        up = SAMPLE_RATE_TELEPHONY // gcd
        down = sr_in // gcd
        audio_8k = scipy.signal.resample_poly(audio_np, up, down)
        audio_8k = np.clip(audio_8k, -1.0, 1.0)

        audio_int16 = (audio_8k * 32767.0).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_g711 = pcm_encode(audio_bytes, self.rtp_pt)

        chunk_size = CHUNK_SIZE_20MS

        start_time = time.perf_counter()
        packet_count = 0

        pt_byte = 0x00
        if self.rtp_pt == 8:
            pt_byte = 0x08

        for i in range(0, len(audio_g711), chunk_size):
            chunk = audio_g711[i: i + chunk_size]
            if len(chunk) < chunk_size:
                pad = b'\xff' if self.rtp_pt == 0 else b'\xd5'
                chunk += pad * (chunk_size - len(chunk))

            self.seq_num = (self.seq_num + 1) % 65536
            self.timestamp = (self.timestamp + 160) % 4294967296
            header = struct.pack('!BBHII', 0x80, pt_byte, self.seq_num, self.timestamp, self.ssrc)

            self.transport.sendto(header + chunk, self.client_addr)

            packet_count += 1
            expected_time = start_time + (packet_count * 0.02)
            delay = expected_time - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

class RTPProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.calls = {}
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        log(f"READY on {RTP_IP}:{RTP_PORT}")

    def datagram_received(self, data, addr):
        if addr not in self.calls:
            self.calls[addr] = CallHandler(self.transport, addr)
            asyncio.create_task(self.calls[addr].send_greeting())
        self.calls[addr].process_incoming_audio(data)

async def main():
    load_models()
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(lambda: RTPProtocol(), local_addr=(RTP_IP, RTP_PORT))
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        transport.close()

if __name__ == "__main__":
    asyncio.run(main())
