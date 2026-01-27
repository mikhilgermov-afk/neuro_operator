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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__).info
error_log = logging.getLogger(__name__).error

log("Starting Voice Bot service...")

try:
    from faster_whisper import WhisperModel
    from openai import OpenAI
    from tts_engine import F5TTSWrapper
    import prompts
except ImportError as e:
    error_log(f"CRITICAL ERROR importing libraries: {e}")
    sys.exit(1)

# Конфигурация
RTP_IP = "0.0.0.0"
RTP_PORT = 10000
SAMPLE_RATE_TELEPHONY = 8000
SAMPLE_RATE_WHISPER = 16000
CHUNK_SIZE_20MS = 160 

stt_model = None
llm_client = None
tts_engine = None

def load_models():
    global stt_model, llm_client, tts_engine
    log("Loading models...")
    stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    llm_client = OpenAI(base_url=os.getenv("LLM_API_URL"), api_key="sk-local-key")
    tts_engine = F5TTSWrapper(stt_model=stt_model)
    log("All models loaded.")

class CallHandler:
    def __init__(self, transport, addr):
        self.transport = transport
        self.client_addr = addr
        self.history = []
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.seq_num = 0
        self.timestamp = 0
        self.ssrc = 123456
        self.is_speaking = False
        self.greeting_sent = False
        self.start_time = 0 
        self.packet_count = 0

    async def send_greeting(self):
        if self.greeting_sent: return
        self.greeting_sent = True
        try:
            log("Generating greeting...")
            sr, audio = tts_engine.generate("Алло, да, я вас слушаю.")
            await self.stream_audio_back(audio, sr)
        except Exception as e:
            error_log(f"Greeting error: {e}")

    def process_incoming_audio(self, data):
        if len(data) <= 12: return
        payload = data[12:]
        try:
            pcm_data = audioop.ulaw2lin(payload, 2)
        except Exception:
            return 

        rms = audioop.rms(pcm_data, 2)
        
        # Снизил порог до 200, чтобы слышать тихие звуки
        if rms > 200: 
            self.silence_frames = 0
            self.audio_buffer.extend(pcm_data)
        else:
            self.silence_frames += 1

        # Увеличил ожидание тишины до 75 фреймов (~1.5 секунды)
        # Бот станет "терпеливее" и не будет перебивать
        if self.silence_frames > 75 and len(self.audio_buffer) > 4000: 
            audio_to_process = self.audio_buffer[:]
            self.audio_buffer = bytearray()
            self.silence_frames = 0
            asyncio.create_task(self.handle_turn(audio_to_process))

    async def handle_turn(self, audio_bytes):
        if not audio_bytes or self.is_speaking: return
        self.is_speaking = True
        
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            if len(audio_np) < 100: return
            audio_16k = scipy.signal.resample_poly(audio_np, SAMPLE_RATE_WHISPER, SAMPLE_RATE_TELEPHONY)

            segments, _ = stt_model.transcribe(audio_16k, language="ru", beam_size=1)
            user_text = " ".join([s.text for s in segments]).strip()
            
            if len(user_text) < 2:
                self.is_speaking = False
                return

            log(f"🗣️ User: {user_text}")
            self.history.append({"role": "user", "content": user_text})

            messages = prompts.create_messages(self.history)
            completion = await asyncio.to_thread(
                llm_client.chat.completions.create,
                model="Qwen/Qwen2.5-7B-Instruct", messages=messages, temperature=0.3, max_tokens=100
            )
            bot_text = completion.choices[0].message.content
            
            log(f"🤖 Bot: {bot_text}")
            self.history.append({"role": "assistant", "content": bot_text})

            sr_out, audio_out = await asyncio.to_thread(tts_engine.generate, bot_text)
            await self.stream_audio_back(audio_out, sr_out)

        except Exception as e:
            error_log(f"Error: {e}")
        finally:
            self.is_speaking = False

    async def stream_audio_back(self, audio_np, sr_in):
        if len(audio_np) == 0: 
            error_log("TTS returned empty audio!")
            return

        duration = len(audio_np) / sr_in
        if duration < 0.1:
             error_log(f"Audio too short ({duration:.2f}s). Skipping to avoid noise.")
             return

        max_val = np.max(np.abs(audio_np))
        if max_val > 0.05:
            audio_np = audio_np / max_val * 0.90
        
        gcd = np.gcd(sr_in, SAMPLE_RATE_TELEPHONY)
        up = SAMPLE_RATE_TELEPHONY // gcd
        down = sr_in // gcd
        audio_8k = scipy.signal.resample_poly(audio_np, up, down)
        audio_8k = np.clip(audio_8k, -1.0, 1.0)
        
        audio_int16 = (audio_8k * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_ulaw = audioop.lin2ulaw(audio_bytes, 2)

        log(f"🔊 Sending {len(audio_ulaw)} bytes ({len(audio_ulaw)/8000:.2f} sec)...")
        
        chunk_size = CHUNK_SIZE_20MS
        
        self.start_time = time.perf_counter()
        self.packet_count = 0

        for i in range(0, len(audio_ulaw), chunk_size):
            chunk = audio_ulaw[i : i + chunk_size]
            if len(chunk) < chunk_size: chunk += b'\xff' * (chunk_size - len(chunk))

            self.seq_num = (self.seq_num + 1) % 65536 
            self.timestamp = (self.timestamp + 160) % 4294967296
            header = struct.pack('!BBHII', 0x80, 0x00, self.seq_num, self.timestamp, self.ssrc)
            
            self.transport.sendto(header + chunk, self.client_addr)
            
            self.packet_count += 1
            expected_time = self.start_time + (self.packet_count * 0.02)
            delay = expected_time - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

class RTPProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.calls = {}
        self.transport = None
    def connection_made(self, transport):
        self.transport = transport
        log(f"✅ READY on {RTP_IP}:{RTP_PORT}")
    def datagram_received(self, data, addr):
        if addr not in self.calls:
            self.calls[addr] = CallHandler(self.transport, addr)
            asyncio.create_task(self.calls[addr].send_greeting())
        self.calls[addr].process_incoming_audio(data)

async def main():
    load_models()
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(lambda: RTPProtocol(), local_addr=(RTP_IP, RTP_PORT))
    try: await asyncio.Future()
    except asyncio.CancelledError: pass
    finally: transport.close()

if __name__ == "__main__":
    asyncio.run(main())
