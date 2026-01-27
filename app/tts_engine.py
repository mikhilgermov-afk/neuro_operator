import sys
import os
import torch
import numpy as np
import torchaudio
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
        
        # --- НАСТРОЙКА РЕФЕРЕНСА ---
        original_ref_audio = "/app/ref_audio.wav"
        self.ref_audio = "/app/ref_audio_trimmed.wav" 
        
        # ВАЖНО: Уменьшили до 6.0 секунд для стабильности (убирает "глотание" букв)
        self._prepare_safe_audio(original_ref_audio, self.ref_audio, max_sec=6.0)

        log("Transcribing reference audio to ensure text match...")
        segments, _ = stt_model.transcribe(self.ref_audio, language="ru")
        self.ref_text = " ".join([s.text for s in segments]).strip()
        log(f"✅ Auto-detected Ref Text: '{self.ref_text}'")

        if os.path.exists(self.ref_audio):
            wav, sr = torchaudio.load(self.ref_audio)
            self.ref_len_samples = int(wav.shape[1] * 24000 / sr)
        else:
            self.ref_len_samples = 0

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

    def _prepare_safe_audio(self, src_path, dst_path, max_sec=6.0):
        if not os.path.exists(src_path):
            log(f"[ERROR] Original audio not found: {src_path}")
            return

        try:
            waveform, sr = torchaudio.load(src_path)
            duration = waveform.shape[1] / sr
            
            if duration > max_sec:
                log(f"Trimming audio from {duration:.2f}s to {max_sec}s for stability")
                waveform = waveform[:, :int(max_sec * sr)]
            else:
                log(f"Audio duration {duration:.2f}s is OK")

            torchaudio.save(dst_path, waveform, sr)
        except Exception as e:
            log(f"Ref audio prep failed: {e}")

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
            audio, sample_rate, _ = infer_process(
                self.ref_audio, self.ref_text, text,
                self.ema_model, self.vocoder, mel_spec_type="vocos",
                device=self.device, nfe_step=16, speed=1.0
            )
            
            if isinstance(audio, torch.Tensor):
                audio = audio.float().cpu().numpy()
            
            total_len = len(audio)
            cut_len = self.ref_len_samples
            
            # Если генерация слишком короткая (сбой), отдаем всё что есть
            if total_len <= cut_len:
                log(f"[WARNING] Gen too short ({total_len} <= {cut_len}). Returning full audio.")
                return 24000, audio

            audio = audio[cut_len:]

            if len(audio) > 0:
                fade_len = min(500, len(audio))
                fade_curve = np.linspace(0, 1, fade_len)
                audio[:fade_len] *= fade_curve

            return 24000, audio 
        except Exception as e:
            log(f"[ERROR] Generation failed: {e}")
            return 24000, np.zeros(0, dtype=np.float32)
