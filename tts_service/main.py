import io
import os
import pathlib
import subprocess
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from huggingface_hub import snapshot_download

from f5_tts.infer.utils_infer import load_vocoder, load_model
from f5_tts.infer.infer_cli import infer_process

try:
    from ruaccent import RUAccent
except Exception:
    RUAccent = None


class TTSRequest(BaseModel):
    text: Optional[str] = None
    input: Optional[str] = None
    out_format: str = "wav"
    response_format: Optional[str] = None
    voice: Optional[str] = None
    speed: Optional[float] = None


app = FastAPI()

MODEL_OBJ = None
MODEL_CFG = None
VOCODER = None
ACCENTIZER = None

TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda")
TTS_MODEL_REPO = os.getenv("TTS_MODEL_REPO", "").strip()
TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", "/models/tts")
TTS_REF_AUDIO = os.getenv("TTS_REF_AUDIO", "/refs/ref_audio.wav")
TTS_REF_TEXT = os.getenv("TTS_REF_TEXT", "/refs/ref_audio.txt")

RUACCENT_ENABLED = os.getenv("RUACCENT_ENABLED", "0") == "1"
RUACCENT_DEVICE = os.getenv("RUACCENT_DEVICE", "CPU")
RUACCENT_WORKDIR = os.getenv("RUACCENT_WORKDIR", "/models/ruaccent")
RUACCENT_MODEL_SIZE = os.getenv("RUACCENT_MODEL_SIZE", "turbo3.1")
RUACCENT_USE_DICT = os.getenv("RUACCENT_USE_DICT", "1") == "1"
RUACCENT_TINY = os.getenv("RUACCENT_TINY", "1") == "1"


def _find_first(root: pathlib.Path, names):
    for n in names:
        p = root / n
        if p.exists():
            return p
    for p in root.rglob("*"):
        if p.is_file() and p.name in names:
            return p
    return None


def _find_best_safetensors(root: pathlib.Path):
    cands = [p for p in root.rglob("*.safetensors") if p.is_file()]
    if not cands:
        return None
    cands.sort(key=lambda p: (p.stat().st_size, str(p)), reverse=True)
    return cands[0]


def _read_ref_text(path: str) -> str:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _accentize(text: str) -> str:
    if not RUACCENT_ENABLED or ACCENTIZER is None:
        return text
    if "+" in text:
        return text
    try:
        return ACCENTIZER.process_all(text)
    except Exception:
        return text


def _wav_bytes_to_format(wav_bytes: bytes, fmt: str) -> bytes:
    fmt = fmt.lower().strip()
    if fmt in ("wav", "wave"):
        return wav_bytes
    if fmt in ("mp3",):
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
            input=wav_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode("utf-8", errors="ignore")[:500])
        return p.stdout
    raise ValueError("unsupported format")


@app.on_event("startup")
def startup():
    global MODEL_OBJ, MODEL_CFG, VOCODER, ACCENTIZER

    model_dir = pathlib.Path(TTS_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    if TTS_MODEL_REPO:
        snapshot_download(
            repo_id=TTS_MODEL_REPO,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        )

    ckpt_file = _find_best_safetensors(model_dir)
    vocab_file = _find_first(model_dir, ["vocab.txt"])
    model_cfg_file = _find_first(model_dir, ["model_cfg.json", "config.json"])

    if ckpt_file is None or vocab_file is None or model_cfg_file is None:
        raise RuntimeError(
            f"model files not found in {model_dir} (ckpt={ckpt_file}, vocab={vocab_file}, cfg={model_cfg_file})"
        )

    MODEL_OBJ, MODEL_CFG = load_model(
        ckpt_file=str(ckpt_file),
        vocab_file=str(vocab_file),
        model_cfg_file=str(model_cfg_file),
        device=TTS_DEVICE,
    )
    VOCODER = load_vocoder(vocoder_name="vocos", device=TTS_DEVICE)

    if RUACCENT_ENABLED and RUAccent is not None:
        ACCENTIZER = RUAccent(workdir=RUACCENT_WORKDIR)
        ACCENTIZER.load(
            omograph_model_size=RUACCENT_MODEL_SIZE,
            use_dictionary=RUACCENT_USE_DICT,
            tiny_mode=RUACCENT_TINY,
            device=RUACCENT_DEVICE,
        )


@app.post("/tts")
@app.post("/v1/audio/speech")
def tts(req: TTSRequest):
    text = (req.text or req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    out_fmt = (req.response_format or req.out_format or "wav").lower().strip()

    ref_text = _read_ref_text(TTS_REF_TEXT)
    gen_text = _accentize(text)

    try:
        audio_np, sr, _ = infer_process(
            ref_audio=TTS_REF_AUDIO,
            ref_text=ref_text,
            gen_text=gen_text,
            model_obj=MODEL_OBJ,
            model_cfg=MODEL_CFG,
            vocoder=VOCODER
        )
        audio_np = np.asarray(audio_np, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()
        out_bytes = _wav_bytes_to_format(wav_bytes, out_fmt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:500])

    media = "audio/wav" if out_fmt == "wav" else "audio/mpeg"
    return Response(content=out_bytes, media_type=media)
