import os
import io
import subprocess
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from pathlib import Path

from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process

try:
    from ruaccent import RUAccent
except Exception:
    RUAccent = None


app = FastAPI()

MODEL_OBJ = None
MODEL_CFG = None
VOCODER = None
ACCENTIZER = None

TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda")
TTS_MODEL_REPO = os.getenv("TTS_MODEL_REPO", "").strip()
TTS_MODEL_DIR = os.getenv("TTS_MODEL_DIR", "/models/f5").strip()

TTS_REF_AUDIO = os.getenv("TTS_REF_AUDIO", "/refs/ref_audio.wav").strip()
TTS_REF_TEXT = os.getenv("TTS_REF_TEXT", "/refs/ref_audio.txt").strip()

RUACCENT_ENABLED = os.getenv("RUACCENT_ENABLED", "0").strip() == "1"
RUACCENT_WORKDIR = os.getenv("RUACCENT_WORKDIR", "/models/ruaccent").strip()
RUACCENT_MODEL_SIZE = os.getenv("RUACCENT_MODEL_SIZE", "turbo3").strip()
RUACCENT_USE_DICT = os.getenv("RUACCENT_USE_DICT", "1").strip() == "1"
RUACCENT_TINY = os.getenv("RUACCENT_TINY", "0").strip() == "1"
RUACCENT_DEVICE = os.getenv("RUACCENT_DEVICE", "cpu").strip()


class TTSRequest(BaseModel):
    text: str | None = None
    input: str | None = None
    response_format: str | None = None
    out_format: str | None = None


def _find_best_safetensors(model_dir: str) -> Path | None:
    p = Path(model_dir)
    if not p.exists():
        return None
    items = sorted(p.glob("*.safetensors"))
    if not items:
        return None
    for name in ["model.safetensors", "ckpt.safetensors", "checkpoint.safetensors"]:
        for it in items:
            if it.name == name:
                return it
    return items[-1]


def _find_first(model_dir: str, names: list[str]) -> Path | None:
    p = Path(model_dir)
    for n in names:
        cand = p / n
        if cand.exists():
            return cand
    for n in names:
        found = list(p.rglob(n))
        if found:
            return found[0]
    return None


def _read_ref_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _accentize(text: str) -> str:
    global ACCENTIZER
    if not RUACCENT_ENABLED or ACCENTIZER is None:
        return text
    try:
        return ACCENTIZER.process_all(text)
    except Exception:
        return text


def _wav_bytes_to_format(wav_bytes: bytes, out_fmt: str) -> bytes:
    out_fmt = (out_fmt or "wav").lower().strip()
    if out_fmt in ("wav", "wave"):
        return wav_bytes
    if out_fmt in ("mp3", "mpeg3"):
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
            input=wav_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode("utf-8", "ignore")[:500])
        return p.stdout
    return wav_bytes


@app.on_event("startup")
def startup_event():
    global MODEL_OBJ, MODEL_CFG, VOCODER, ACCENTIZER

    model_dir = TTS_MODEL_DIR

    if TTS_MODEL_REPO:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=TTS_MODEL_REPO,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception:
            pass

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


@app.get("/health")
def health():
    ok = MODEL_OBJ is not None and MODEL_CFG is not None and VOCODER is not None
    return {"ok": bool(ok)}


@app.post("/tts")
@app.post("/v1/audio/speech")
def tts(req: TTSRequest):
    text = (req.text or req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    if MODEL_OBJ is None or MODEL_CFG is None or VOCODER is None:
        raise HTTPException(status_code=503, detail="model not ready")

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
            vocoder=VOCODER,
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
