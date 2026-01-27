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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__).info
error_log = logging.getLogger(__name__).error

log("Starting Voice Bot service...")

try:
    log("Importing libraries...")
    from faster_whisper import WhisperModel
    from openai import OpenAI
    from tts_engine import F5TTSWrapper
    import prompts
    log("Libraries imported.")
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
    tts_engine = F5TTSWrapper()
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

    async def send_greeting(self):
        if self.greeting_sent: return
        self.greeting_sent = True
        try:
            # Короткое приветствие работает лучше
            sr, audio = tts_engine.generate("Да, я слушаю.")
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
        
        # Порог VAD (300)
        if rms > 300: 
            self.silence_frames = 0
            self.audio_buffer.extend(pcm_data)
        else:
            self.silence_frames += 1

        # Ждем ~1 секунду тишины
        if self.silence_frames > 50 and len(self.audio_buffer) > 4000:
            audio_to_process = self.audio_buffer[:]
            self.audio_buffer = bytearray()
            self.silence_frames = 0
            asyncio.create_task(self.handle_turn(audio_to_process))

    async def handle_turn(self, audio_bytes):
        if not audio_bytes or self.is_speaking: return
        self.is_speaking = True
        
        try:
            # 1. Байты -> Float32
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 2. Ресемплинг для Whisper (8k -> 16k)
            # Используем resample_poly для качества
            if len(audio_np) < 100: return
            audio_16k = scipy.signal.resample_poly(audio_np, SAMPLE_RATE_WHISPER, SAMPLE_RATE_TELEPHONY)

            # 3. STT
            segments, _ = stt_model.transcribe(audio_16k, language="ru", beam_size=1)
            user_text = " ".join([s.text for s in segments]).strip()
            
            if len(user_text) < 2:
                self.is_speaking = False
                return

            log(f"🗣️ User: {user_text}")
            self.history.append({"role": "user", "content": user_text})

            # 4. LLM
            messages = prompts.create_messages(self.history)
            completion = await asyncio.to_thread(
                llm_client.chat.completions.create,
                model="Qwen/Qwen2.5-7B-Instruct", messages=messages, temperature=0.6, max_tokens=150
            )
            bot_text = completion.choices[0].message.content
            log(f"🤖 Bot: {bot_text}")
            self.history.append({"role": "assistant", "content": bot_text})

            # 5. TTS
            sr_out, audio_out = await asyncio.to_thread(tts_engine.generate, bot_text)
            
            # 6. Отправка
            await self.stream_audio_back(audio_out, sr_out)

        except Exception as e:
            error_log(f"Error: {e}")
        finally:
            self.is_speaking = False

    async def stream_audio_back(self, audio_np, sr_in):
        if len(audio_np) == 0: return

        # --- 1. Умная нормализация ---
        # Поднимаем громкость, только если сигнал не пустой шум
        max_val = np.max(np.abs(audio_np))
        if max_val > 0.05: # Если есть хоть какой-то голос
            # Нормализуем до 90% (оставляем запас от клиппинга)
            audio_np = audio_np / max_val * 0.90
        
        # --- 2. Качественный ресемплинг (24k -> 8k) ---
        # resample_poly убирает "трубный" звон (aliasing)
        # up=1, down=3 (24000 * 1 / 3 = 8000)
        # Если sr_in=24000, то up=1, down=3.
        # Вычисляем НОД для произвольных частот:
        gcd = np.gcd(sr_in, SAMPLE_RATE_TELEPHONY)
        up = SAMPLE_RATE_TELEPHONY // gcd
        down = sr_in // gcd
        
        audio_8k = scipy.signal.resample_poly(audio_np, up, down)
        
        # --- 3. Клиппинг ---
        # Жестко срезаем пики, которые могли возникнуть при фильтрации
        audio_8k = np.clip(audio_8k, -1.0, 1.0)
        
        # --- 4. Кодирование ---
        audio_int16 = (audio_8k * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_ulaw = audioop.lin2ulaw(audio_bytes, 2)

        log(f"🔊 Sending {len(audio_ulaw)} bytes...")
        
        chunk_size = CHUNK_SIZE_20MS
        for i in range(0, len(audio_ulaw), chunk_size):
            chunk = audio_ulaw[i : i + chunk_size]
            if len(chunk) < chunk_size: chunk += b'\xff' * (chunk_size - len(chunk))

            self.seq_num = (self.seq_num + 1) % 65536 
            self.timestamp = (self.timestamp + 160) % 4294967296
            header = struct.pack('!BBHII', 0x80, 0x00, self.seq_num, self.timestamp, self.ssrc)
            
            self.transport.sendto(header + chunk, self.client_addr)
            await asyncio.sleep(0.0195) 

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
