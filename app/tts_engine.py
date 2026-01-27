import sys
import os
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

os.environ["HF_HOME"] = os.getenv("HF_CACHE_DIR", "/models/hf")

try:
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
except ImportError as e:
    print(f"[FATAL] Could not import F5-TTS library: {e}", flush=True)
    sys.exit(1)

def log(msg):
    print(f"[TTS-ENGINE] {msg}", flush=True)

def get_sinusoidal_pe(pos_len: int, dim: int):
    pe = torch.zeros(pos_len, dim)
    position = torch.arange(0, pos_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class F5TTSWrapper:
    def __init__(self, stt_model, repo_id="Misha24-10/F5-TTS_RUSSIAN", device="cuda"):
        self.device = device
        self.repo_id = repo_id
        
        log(f"Initializing F5-TTS on device: {self.device}")

        try:
            self.model_path = snapshot_download(repo_id)
        except Exception as e:
            log(f"[ERROR] Download failed: {e}")
            raise e

        self.checkpoint_file = self._find_file(self.model_path, [".pt", ".safetensors"], "model_")
        self.vocab_file = self._find_file(self.model_path, ["vocab.txt"])
        
        # --- ПОДГОТОВКА С ПАУЗОЙ ---
        self.ref_audio_path = "/app/ref_audio.wav"
        self.temp_ref_path = "/app/ref_audio_padded.wav"
        
        # 1. Загружаем, режем до 3с и добавляем 1с ТИШИНЫ
        self.ref_audio_tensor, self.cut_point_samples = self._prepare_padded_reference(
            self.ref_audio_path, 
            self.temp_ref_path, 
            max_sec=3.0, 
            pad_sec=1.0 # <--- Важная добавка
        )
        
        # 2. Авто-текст
        log("Transcribing reference...")
        segments, _ = stt_model.transcribe(self.temp_ref_path, language="ru")
        self.ref_text = " ".join([s.text for s in segments]).strip()
        if len(self.ref_text) < 2: 
            self.ref_text = "Пример голоса."
        
        log(f"✅ Ref Text: '{self.ref_text}'")
        log(f"✅ Cut point set at sample: {self.cut_point_samples}")

        log("Loading F5-TTS models...")
        self.vocoder = load_vocoder(is_local=False)
        
        model_cfg = OmegaConf.create({
            "model": {
                "backbone": "DiT",
                "arch": {
                    "dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, 
                    "text_dim": 512, "conv_layers": 4
                }
            }
        })

        self.ema_model = load_model(
            model_cls=DiT,
            model_cfg=model_cfg.model.arch,
            ckpt_path=self.checkpoint_file,
            mel_spec_type="vocos",
            vocab_file=self.vocab_file,
            device=self.device
        )
        self.ema_model = self.ema_model.to(torch.float32)
        self._patch_sinusoidal_memory(self.ema_model, target_len=32000)
        log("✅ F5-TTS Engine Ready.")

    def _prepare_padded_reference(self, src, dst, max_sec=3.0, pad_sec=1.0):
        """Создает референс: Аудио(3с) + Тишина(1с)"""
        if not os.path.exists(src):
            log(f"[ERROR] {src} not found!")
            return torch.zeros(1, 24000), 0

        waveform, sr = torchaudio.load(src)
        
        # Mono + 24k
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 24000: waveform = T.Resample(sr, 24000)(waveform)
        
        # Обрезаем сам голос до 3 сек
        voice_len = int(24000 * max_sec)
        if waveform.shape[1] > voice_len:
            waveform = waveform[:, :voice_len]
        
        # Добавляем тишину (pad)
        pad_len = int(24000 * pad_sec)
        silence = torch.zeros(1, pad_len)
        padded_waveform = torch.cat([waveform, silence], dim=1)

        # Сохраняем
        torchaudio.save(dst, padded_waveform, 24000)
        
        # Точка обрезки = длина голоса + половина тишины (чтобы наверняка)
        cut_point = waveform.shape[1] + int(pad_len * 0.7) 
        
        return padded_waveform, cut_point

    def _patch_sinusoidal_memory(self, module, target_len=32000):
        for name, submod in module.named_modules():
            freqs = getattr(submod, "freqs_cis", None)
            if freqs is None and hasattr(submod, "_buffers"):
                freqs = submod._buffers.get("freqs_cis")
            if freqs is not None and freqs.shape[0] < target_len:
                dim = freqs.shape[-1]
                new_pe = get_sinusoidal_pe(target_len, dim).to(device=self.device, dtype=torch.float32)
                if hasattr(submod, "freqs_cis"): submod.freqs_cis = new_pe
                if hasattr(submod, "_buffers"): submod._buffers["freqs_cis"] = new_pe

    def _find_file(self, path, extensions, substring=None):
        for root, dirs, files in os.walk(path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    if substring and substring not in file: continue
                    return os.path.join(root, file)
        return None

    def generate(self, text):
        if not text: return 24000, np.zeros(0, dtype=np.float32)
        
        try:
            # Генерация (speed 1.0 для качества)
            audio, sample_rate, _ = infer_process(
                self.temp_ref_path, self.ref_text, text,
                self.ema_model, self.vocoder, mel_spec_type="vocos",
                device=self.device, nfe_step=32, speed=1.0
            )
            
            if isinstance(audio, torch.Tensor):
                audio = audio.float().cpu().numpy()
            
            # --- БЕЗОПАСНАЯ ОБРЕЗКА ---
            cut_len = self.cut_point_samples
            
            if len(audio) > cut_len:
                audio = audio[cut_len:]
            else:
                # Если сгенерировалось меньше, чем длина (голос+пауза), значит бот промолчал
                log(f"[WARN] Output too short ({len(audio)} <= {cut_len}). Returning silence.")
                return 24000, np.zeros(0, dtype=np.float32)

            # Fade-in
            if len(audio) > 0:
                fade_len = min(500, len(audio))
                fade = np.linspace(0, 1, fade_len)
                audio[:fade_len] *= fade

            return 24000, audio 
        except Exception as e:
            log(f"[ERROR] Generation failed: {e}")
            return 24000, np.zeros(0, dtype=np.float32)
