#!/usr/bin/env bash
set -euo pipefail

if [ "${WARMUP_MODELS:-0}" = "1" ]; then
python - <<'PY'
import os
from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
repos = [os.getenv("TTS_MODEL_REPO"), os.getenv("WHISPER_MODEL_REPO")]
for repo in [r for r in repos if r]:
    snapshot_download(repo_id=repo, token=token)
PY
fi

if [ $# -eq 0 ]; then
exec python -u /app/main.py
fi

exec "$@"
