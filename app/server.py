#! /usr/bin/env python3
from flask import Flask, request, jsonify, Response
import logging
import os
import yaml
import torch
import soundfile as sf
import numpy as np
import sys
import socket
import time
from threading import Thread
from omegaconf import OmegaConf
from hydra.utils import get_class

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("F5-TTS-Server")

app = Flask(__name__)

# --- КОНФИГУРАЦИЯ ---
OUTPUT_FOLDER = 'output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Глобальные переменные
ema_model = None
vocoder = None

# Пытаемся импортировать утилиты F5-TTS
try:
    from f5_tts.infer.utils_infer import (
        device, mel_spec_type, target_rms, cross_fade_duration,
        nfe_step, cfg_strength, sway_sampling_coef, speed,
        fix_duration, infer_process, load_model, load_vocoder,
        preprocess_ref_audio_text
    )
    logger.info("Библиотека f5_tts успешно импортирована.")
except ImportError as e:
    logger.error(f"Ошибка импорта f5_tts: {e}")
    sys.exit(1)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def is_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False

def get_samples_config():
    """Загружает конфиг сэмпла или возвращает дефолтный, чтобы не падать"""
    samples_path = 'samples/samples.yaml'
    if os.path.exists(samples_path):
        with open(samples_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # Дефолтный конфиг, если файла нет (ищем референс в папке app)
        return {
            "default": {
                "ref_audio": "ref_audio.wav",
                "ref_text": "Тестовый референс."
            }
        }

def initialize_model():
    global ema_model, vocoder
    
    # ПУТИ К МОДЕЛЯМ (Настрой под свою структуру папок!)
    # Если ты используешь Docker из прошлого примера, пути могут отличаться.
    # Здесь пример для локальной HF загрузки:
    
    # Вариант 1: Жесткие пути (как в твоем коде)
    # ckpt_path = 'f5-tts-russian/F5TTS_v1_Base_v2/model_last_inference.safetensors'
    # vocab_file = 'ckpts/ru_f5tts/F5TTS_v1_Base/vocab.txt'
    # model_cfg_path = 'ckpts/ru_f5tts/F5TTS_v1_Base_v2/F5TTS_v1_Base.yaml'
    
    # Вариант 2: Пути из кэша HuggingFace (если скачивали скриптом)
    # Для простоты используем заглушки, библиотека сама найдет если скачать через snapshot_download
    # Но раз у тебя уже есть пути, используем их.
    
    # ВАЖНО: В Docker'е мы использовали snapshot_download. 
    # Давай сделаем универсально:
    try:
        from huggingface_hub import snapshot_download
        repo_id = "Misha24-10/F5-TTS_RUSSIAN"
        logger.info(f"Проверка/Загрузка модели {repo_id}...")
        model_path = snapshot_download(repo_id)
        
        # Ищем файлы внутри скачанного
        def find_file(name):
            for root, _, files in os.walk(model_path):
                if name in files: return os.path.join(root, name)
            return None

        ckpt_path = find_file("model_1200000.pt") or find_file("model_last.pt")
        # Если не нашли по имени, ищем любой safetensors/pt
        if not ckpt_path:
             for root, _, files in os.walk(model_path):
                for f in files:
                    if f.endswith(".safetensors") or f.endswith(".pt"):
                        ckpt_path = os.path.join(root, f)
                        break
        
        vocab_file = find_file("vocab.txt")
        # Конфиг берем дефолтный из библиотеки, если нет yaml
        model_cfg = None 
        
        logger.info(f"Найден чекпоинт: {ckpt_path}")
        logger.info(f"Найден словарь: {vocab_file}")

    except Exception as e:
        logger.error(f"Ошибка автопоиска модели: {e}")
        return False

    # Настройки Vocoder (Vocos)
    vocoder_name = "vocos"
    
    try:
        logger.info("--- ЗАГРУЗКА МОДЕЛЕЙ ---")
        current_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Vocoder
        logger.info(f"Loading Vocoder: {vocoder_name}")
        vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, device=current_device)
        
        # 2. TTS Model
        logger.info(f"Loading TTS Model...")
        
        # Конфигурация архитектуры (стандартная для F5-Base)
        model_cfg_dict = dict(
            model=dict(
                backbone="DiT",
                arch=dict(
                    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
                )
            )
        )
        model_cfg = OmegaConf.create(model_cfg_dict)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        ema_model = load_model(
            model_cls, 
            model_arc, 
            ckpt_path, 
            mel_spec_type=vocoder_name, 
            vocab_file=vocab_file, 
            device=current_device
        )
        
        logger.info("✅ Модели успешно загружены!")
        return True
        
    except Exception as e:
        logger.exception("CRITICAL ERROR loading models")
        return False

# --- МАРШРУТЫ (ROUTES) ---

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния сервера"""
    status = 'ok' if ema_model is not None and vocoder is not None else 'error'
    return jsonify({
        'status': status, 
        'device': str(device)
    })

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """Генерация речи"""
    global ema_model, vocoder
    
    if not ema_model or not vocoder:
        return jsonify({'error': 'Models not loaded'}), 503
    
    try:
        # Данные из формы или JSON
        if request.is_json:
            data = request.json
        else:
            data = request.form

        gen_text = data.get('text', '').strip()
        sample_name = data.get('sample', 'default') # Если не указан, берем дефолт
        
        if not gen_text:
            return jsonify({'error': 'No text provided'}), 400

        # Конфиг сэмплов
        samples_config = get_samples_config()
        if sample_name not in samples_config:
            # Фолбэк на первый попавшийся или ошибку
            sample_name = list(samples_config.keys())[0]
            
        sample_data = samples_config[sample_name]
        ref_audio_path = sample_data.get('ref_audio')
        ref_text_orig = sample_data.get('ref_text', "")

        # Проверяем путь к референсу
        if not os.path.exists(ref_audio_path):
             # Пробуем искать в текущей папке
             if os.path.exists(os.path.join("app", ref_audio_path)):
                 ref_audio_path = os.path.join("app", ref_audio_path)
             else:
                 return jsonify({'error': f'Ref audio not found: {ref_audio_path}'}), 400

        # Препроцессинг
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text_orig)
        
        # Инференс
        logger.info(f"Генерация: '{gen_text}' (ref: {sample_name})")
        
        audio_segment, final_sample_rate, _ = infer_process(
            ref_audio, ref_text, gen_text,
            ema_model, vocoder,
            mel_spec_type=mel_spec_type, target_rms=target_rms,
            cross_fade_duration=cross_fade_duration, nfe_step=nfe_step,
            cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef,
            speed=speed, fix_duration=fix_duration, device=device
        )

        # Конвертация в байты для отправки (WAV)
        import io
        byte_io = io.BytesIO()
        sf.write(byte_io, audio_segment, final_sample_rate, format='WAV')
        byte_io.seek(0)
        
        return Response(byte_io, mimetype="audio/wav")

    except Exception as e:
        logger.exception("Ошибка синтеза")
        return jsonify({'error': str(e)}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Мягкая остановка сервера"""
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
        return jsonify({'message': 'Server shutting down'}), 200
    return jsonify({'error': 'Not running with Werkzeug'}), 500

if __name__ == '__main__':
    port = 3000
    if not is_port_free(port):
        logger.error(f"Порт {port} занят!")
        sys.exit(1)
        
    if initialize_model():
        logger.info(f"🚀 Сервер запущен на порту {port}")
        # threaded=True важно для одновременной обработки, но infer_process блокирующий
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
