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
SAMPLE_RATE_WHISPER = 16000 # Whisper ожидает 16k
CHUNK_SIZE_20MS = 160 

# Глобальные модели
stt_model = None
llm_client = None
tts_engine = None

def load_models():
    global stt_model, llm_client, tts_engine
    
    log("1/3 Loading Whisper (STT)...")
    # medium или large-v3-turbo - выбирай по видеопамяти
    stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    log("   > Whisper loaded.")

    log("2/3 Connecting to LLM Client...")
    llm_client = OpenAI(base_url=os.getenv("LLM_API_URL"), api_key="sk-local-key")
    log("   > LLM Client configured.")

    log("3/3 Loading F5-TTS (Voice)...")
    tts_engine = F5TTSWrapper()
    log("   > F5-TTS loaded.")

class CallHandler:
    def __init__(self, transport, addr):
        self.transport = transport
        self.client_addr = addr
        self.history = []
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        
        # RTP State
        self.seq_num = 0
        self.timestamp = 0
        self.ssrc = 123456
        
        self.is_speaking = False
        self.greeting_sent = False

    async def send_greeting(self):
        """Отправляет приветствие при начале звонка"""
        if self.greeting_sent: return
        self.greeting_sent = True
        
        greeting_text = "Здравствуйте! Я слушаю."
        log(f"������ Sending greeting: {greeting_text}")
        
        # Генерируем фейковый "User" ход, чтобы запустить цепочку, 
        # или просто генерируем TTS напрямую. Лучше напрямую.
        try:
            sr, audio = tts_engine.generate(greeting_text)
            await self.stream_audio_back(audio, sr)
        except Exception as e:
            error_log(f"Greeting failed: {e}")

    def process_incoming_audio(self, data):
        # Удаляем RTP заголовок (12 байт)
        if len(data) <= 12: return
        payload = data[12:]
        
        try:
            # ulaw -> pcm16
            pcm_data = audioop.ulaw2lin(payload, 2)
        except Exception:
            return 

        # VAD (детектор голоса по энергии)
        rms = audioop.rms(pcm_data, 2)
        
        # Порог (подбирать экспериментально, 100-300 обычно ок для тишины)
        if rms > 150: 
            self.silence_frames = 0
            self.audio_buffer.extend(pcm_data)
        else:
            self.silence_frames += 1

        # Если накопили буфер и наступила тишина (0.6 сек)
        if self.silence_frames > 30 and len(self.audio_buffer) > 4000:
            # Запускаем обработку в фоне, чтобы не блочить прием пакетов
            asyncio.create_task(self.handle_turn())
            self.audio_buffer = bytearray()
            self.silence_frames = 0

    async def handle_turn(self):
        if self.is_speaking: return # Не перебиваем сами себя пока (простая логика)
        
        buffer_len = len(self.audio_buffer)
        log(f"--- Processing Turn (Buffer: {buffer_len} bytes) ---")
        self.is_speaking = True
        
        try:
            # 1. Подготовка аудио для Whisper
            # Конвертация bytearray -> numpy float32
            audio_np_8k = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
            # --- ВАЖНО: РЕСЕМПЛИНГ 8k -> 16k ---
            # Whisper требует 16kHz. 
            num_samples_16k = int(len(audio_np_8k) * SAMPLE_RATE_WHISPER / SAMPLE_RATE_TELEPHONY)
            audio_np_16k = scipy.signal.resample(audio_np_8k, num_samples_16k)
            # -----------------------------------

            # 2. Whisper
            segments, _ = stt_model.transcribe(audio_np_16k, language="ru", beam_size=1)
            user_text = " ".join([s.text for s in segments]).strip()
            
            if not user_text or len(user_text) < 2:
                log("   > Silence/Noise detected (No text)")
                self.is_speaking = False
                return

            log(f"������️ User: {user_text}")
            self.history.append({"role": "user", "content": user_text})

            # 3. LLM
            messages = prompts.create_messages(self.history)
            log("   > Asking LLM...")
            
            # Используем to_thread, чтобы requests не блокировал event loop
            completion = await asyncio.to_thread(
                llm_client.chat.completions.create,
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=messages,
                temperature=0.6,
                max_tokens=150
            )
            
            bot_text = completion.choices[0].message.content
            log(f"������ Bot: {bot_text}")
            self.history.append({"role": "assistant", "content": bot_text})

            # 4. TTS
            log("   > Generating Audio...")
            # to_thread для тяжелой генерации
            sr_out, audio_out = await asyncio.to_thread(tts_engine.generate, bot_text)
            
            # 5. Отправка
            await self.stream_audio_back(audio_out, sr_out)
            
        except Exception as e:
            error_log(f"Error processing turn: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_speaking = False

    async def stream_audio_back(self, audio_float, sr_in):
        if len(audio_float) == 0: return

        # Ресемплинг обратно в 8k для телефона
        num_samples_8k = int(len(audio_float) * SAMPLE_RATE_TELEPHONY / sr_in)
        audio_8k = scipy.signal.resample(audio_float, num_samples_8k)
        
        # float32 -> int16 -> ulaw
        audio_int16 = (audio_8k * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        audio_ulaw = audioop.lin2ulaw(audio_bytes, 2)

        chunk_size = CHUNK_SIZE_20MS
        log(f"������ Sending audio back ({len(audio_ulaw)} bytes)...")
        
        for i in range(0, len(audio_ulaw), chunk_size):
            chunk = audio_ulaw[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b'\xff' * (chunk_size - len(chunk)) # Silence padding

            # RTP Header
            # Исправлен wrap seq (65536) и timestamp (2^32)
            self.seq_num = (self.seq_num + 1) % 65536 
            self.timestamp = (self.timestamp + 160) % 4294967296
            
            header = struct.pack('!BBHII', 0x80, 0x00, self.seq_num, self.timestamp, self.ssrc)
            
            # Отправка через asyncio transport (неблокирующая)
            self.transport.sendto(header + chunk, self.client_addr)
            
            # Тайминг 20мс
            await asyncio.sleep(0.0195) 

class RTPProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.calls = {} # addr -> CallHandler
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        log(f"✅ SYSTEM READY. Listening RTP on {RTP_IP}:{RTP_PORT}")

    def datagram_received(self, data, addr):
        if addr not in self.calls:
            log(f"New call from {addr}")
            handler = CallHandler(self.transport, addr)
            self.calls[addr] = handler
            # Приветствие при первом пакете
            asyncio.create_task(handler.send_greeting())
        
        # Передаем данные в обработчик
        self.calls[addr].process_incoming_audio(data)

async def main():
    load_models()
    
    loop = asyncio.get_running_loop()
    
    # Запускаем UDP сервер через DatagramProtocol (Native AsyncIO)
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: RTPProtocol(),
        local_addr=(RTP_IP, RTP_PORT)
    )

    try:
        # Держим сервис живым
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        transport.close()

if __name__ == "__main__":
    asyncio.run(main())