import os
import io
import base64
from pathlib import Path
from contextlib import asynccontextmanager
import importlib.util
import inspect

import torch
import soundfile as sf
import numpy as np

from fastapi import FastAPI, Body, Query
from fastapi.responses import Response, JSONResponse

from cached_path import cached_path

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import load_model, infer_batch_process


_state = {
    "ready": False,
    "device": None,
    "vocoder": None,
    "model": None,
    "vocab_file": None,
    "ref_audio": None,
    "ref_text": None,
}


def _get_device() -> str:
    dev = (os.getenv("TTS_DEVICE") or "").strip().lower()
    if dev:
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_vocab_file() -> str:
    spec = importlib.util.find_spec("f5_tts")
    if spec and spec.submodule_search_locations:
        for loc in list(spec.submodule_search_locations):
            p = Path(str(loc)) / "infer" / "examples" / "vocab.txt"
            if p.exists():
                return str(p)

    candidates = [
        "/opt/conda/lib/python3.11/site-packages/f5_tts/infer/examples/vocab.txt",
        "/opt/conda/lib/python3.10/site-packages/f5_tts/infer/examples/vocab.txt",
        "/usr/local/lib/python3.11/site-packages/f5_tts/infer/examples/vocab.txt",
        "/usr/local/lib/python3.10/dist-packages/f5_tts/infer/examples/vocab.txt",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return ""


def _read_ref_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore").strip()


def _pick_ckpt_file() -> str:
    env_path = (os.getenv("TTS_CKPT_FILE") or os.getenv("TTS_CKPT_PATH") or "").strip()
    if env_path:
        return str(env_path)

    repo = (os.getenv("TTS_MODEL_REPO") or "").strip()
    name = (os.getenv("TTS_CKPT_NAME") or "").strip()

    candidates = []
    if name:
        candidates.append(name)
    candidates.extend([
        "model_1200000.safetensors",
        "model.safetensors",
        "model_1200000.pt",
        "model.pt",
    ])

    if repo:
        for c in candidates:
            try:
                return str(cached_path(f"hf://{repo}/{c}"))
            except Exception:
                pass

    try:
        return str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
    except Exception:
        return str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))


def _model_cfg_and_cls(model_type: str):
    t = (model_type or "F5-TTS").strip()
    if t == "E2-TTS":
        return UNetT, dict(dim=1024, depth=24, heads=16, ff_mult=4)
    return DiT, dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


def _load_vocoder(device: str):
    from vocos import Vocos

    local_path = (os.getenv("TTS_VOCODER_LOCAL_PATH") or "").strip()
    repo = (os.getenv("TTS_VOCODER_REPO") or "charactr/vocos-mel-24khz").strip()

    if local_path:
        vocoder = Vocos.from_pretrained(local_path)
    else:
        vocoder = Vocos.from_pretrained(repo)

    if hasattr(vocoder, "to"):
        vocoder = vocoder.to(device)
    if hasattr(vocoder, "eval"):
        vocoder.eval()
    return vocoder


def _call_load_model(model_cls, model_cfg, ckpt_file: str, vocab_file: str, device: str, ode_method: str, use_ema: bool):
    tok = (os.getenv("TTS_TOKENIZER") or "custom").strip()
    dtype_env = (os.getenv("TTS_DTYPE") or "").strip().lower()

    dtype = None
    if dtype_env in ("fp16", "float16"):
        dtype = torch.float16
    elif dtype_env in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif dtype_env in ("fp32", "float32"):
        dtype = torch.float32

    sig = inspect.signature(load_model)
    params = sig.parameters

    kwargs = {}
    for name in params.keys():
        if name in ("model_cls", "model_class", "model_type", "model"):
            kwargs[name] = model_cls
        elif name in ("model_cfg", "model_config", "cfg", "config"):
            kwargs[name] = model_cfg
        elif name in ("ckpt_path", "ckpt_file", "checkpoint_path", "checkpoint", "ckpt"):
            kwargs[name] = str(ckpt_file)
        elif name in ("vocab_file", "vocab_path", "dataset_name", "vocab"):
            kwargs[name] = str(vocab_file)
        elif name in ("tokenizer", "token_type", "token"):
            kwargs[name] = tok
        elif name in ("device", "dev"):
            kwargs[name] = device
        elif name in ("ode_method", "ode", "solver", "method"):
            kwargs[name] = ode_method
        elif name in ("use_ema", "ema"):
            kwargs[name] = use_ema
        elif name == "dtype" and dtype is not None:
            kwargs[name] = dtype

    required_missing = []
    for name, p in params.items():
        if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            if name not in kwargs:
                required_missing.append(name)

    if required_missing:
        raise RuntimeError("load_model_missing_params:" + ",".join(required_missing))

    return load_model(**kwargs)


def _load_all():
    device = _get_device()
    vocab_file = (os.getenv("TTS_VOCAB_FILE") or "").strip() or _default_vocab_file()
    if not vocab_file or not Path(vocab_file).exists():
        raise RuntimeError("vocab_file_not_found")
    ckpt_file = str(_pick_ckpt_file())

    model_type = (os.getenv("TTS_MODEL_TYPE") or "F5-TTS").strip()
    ode_method = (os.getenv("TTS_ODE_METHOD") or "euler").strip()
    use_ema = (os.getenv("TTS_USE_EMA") or "1").strip() not in ("0", "false", "False")

    model_cls, model_cfg = _model_cfg_and_cls(model_type)
    vocoder = _load_vocoder(device)
    model = _call_load_model(model_cls, model_cfg, ckpt_file, vocab_file, device, ode_method, use_ema)

    ref_audio = (os.getenv("TTS_REF_AUDIO") or "").strip()
    ref_text_path = (os.getenv("TTS_REF_TEXT") or "").strip()
    ref_text = _read_ref_text(ref_text_path) if ref_text_path else ""

    _state["ready"] = True
    _state["device"] = device
    _state["vocoder"] = vocoder
    _state["model"] = model
    _state["vocab_file"] = vocab_file
    _state["ref_audio"] = ref_audio
    _state["ref_text"] = ref_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": _state["ready"], "device": _state["device"]}


@app.post("/tts")
@app.post("/synthesize")
def tts(
    text: str = Body("", embed=True),
    gen_text: str = Body("", embed=True),
    ref_text: str = Body("", embed=True),
    speed: float = Body(1.0, embed=True),
    return_json: int = Query(0),
):
    if not _state["ready"]:
        return JSONResponse({"error": "not_ready"}, status_code=503)

    t = (text or "").strip() or (gen_text or "").strip()
    if not t:
        return JSONResponse({"error": "empty_text"}, status_code=400)

    ra = (_state["ref_audio"] or "").strip()
    if not ra or not Path(ra).exists():
        return JSONResponse({"error": "ref_audio_missing"}, status_code=500)

    rt = (ref_text or "").strip() or (_state["ref_text"] or "").strip()
    if not rt:
        return JSONResponse({"error": "ref_text_missing"}, status_code=500)

    ref_audio = _state.get("ref_audio_tuple")

    ref_audio_path = _state.get("ref_audio_path_cached")

    if ref_audio is None or ref_audio_path != ra:

        x, sr = sf.read(ra, dtype="float32", always_2d=True)

        x = x.T

        ref_audio = torch.from_numpy(x)

        _state["ref_audio_tuple"] = (ref_audio, int(sr))

        _state["ref_audio_path_cached"] = ra

    ref_audio = _state["ref_audio_tuple"]
    waves = infer_batch_process(
        ref_audio,
        rt,
        [t],
        _state["model"],
        _state["vocoder"],
        mel_spec_type="vocos",
        speed=float(speed),
        device=_state["device"],
    )

    wav_obj = next(waves)

    out_sr = 24000

    if isinstance(wav_obj, (tuple, list)):

        if len(wav_obj) >= 2 and isinstance(wav_obj[1], (int, float)):

            out_sr = int(wav_obj[1])

        wav_obj = wav_obj[0]

    elif isinstance(wav_obj, dict):

        for k in ("wav", "audio", "audio_np", "audio_wave", "samples"):

            if k in wav_obj:

                wav_obj = wav_obj[k]

                break

    if hasattr(wav_obj, "detach"):

        wav = wav_obj.detach().cpu().numpy()

    else:

        wav = wav_obj

    wav = np.asarray(wav)

    if wav.ndim == 2:

        if wav.shape[0] == 1:

            wav = wav[0]

        elif wav.shape[1] == 1:

            wav = wav[:, 0]

    buf = io.BytesIO()

    sf.write(buf, wav, out_sr, format="WAV", subtype="PCM_16")

    data = buf.getvalue()

    if int(return_json) == 1:
        b64 = base64.b64encode(data).decode("ascii")
        return {"sample_rate": 24000, "audio_wav_base64": b64}

    return Response(content=data, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    host = (os.getenv("TTS_HOST") or "0.0.0.0").strip()
    port = int((os.getenv("TTS_PORT") or "10002").strip())
    uvicorn.run(app, host=host, port=port, log_level="info")
