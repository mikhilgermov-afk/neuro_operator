import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

def write_file(path: Path, text: str):
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")

def ensure_path_import(src: str) -> str:
    if re.search(r"^\s*from\s+pathlib\s+import\s+Path\s*$", src, re.M):
        return src
    lines = src.splitlines(True)
    last_import_idx = -1
    for i, line in enumerate(lines[:200]):
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            last_import_idx = i
        elif last_import_idx != -1 and s and not s.startswith(("import ", "from ")):
            break
    ins = "from pathlib import Path\n"
    if last_import_idx == -1:
        return ins + src
    lines.insert(last_import_idx + 1, ins)
    return "".join(lines)

def patch_main_py(path: Path):
    src = path.read_text(encoding="utf-8")
    if 'ASR_MODEL = os.getenv("ASR_MODEL", "large-v3-turbo")' not in src:
        return False

    src = ensure_path_import(src)

    block = (
        'ASR_MODEL = os.getenv("ASR_MODEL", "large-v3-turbo")\n'
        '\n'
        'if ASR_MODEL == "large-v3-turbo":\n'
        '    _snapshots = (Path(__file__).resolve().parents[1] / "models" / "voice" / "hf" / "hub" / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo" / "snapshots")\n'
        '    if _snapshots.is_dir():\n'
        '        _dirs = sorted([p for p in _snapshots.iterdir() if p.is_dir()])\n'
        '        if _dirs:\n'
        '            ASR_MODEL = str(_dirs[-1])\n'
        '        else:\n'
        '            ASR_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"\n'
        '    else:\n'
        '        ASR_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"\n'
    )

    src2 = src.replace('ASR_MODEL = os.getenv("ASR_MODEL", "large-v3-turbo")\n', block, 1)
    if src2 == src:
        return False
    write_file(path, src2)
    return True

def patch_tts_service_py(path: Path):
    src = path.read_text(encoding="utf-8")
    if "ckpt_path=str(ckpt_path)" in src:
        return False
    src2 = re.sub(r"ckpt_path\s*=\s*ckpt_path\b", "ckpt_path=str(ckpt_path)", src)
    if src2 == src:
        src2 = re.sub(r"ckpt_path\s*=\s*Path\((.*?)\)", r"ckpt_path=str(Path(\1))", src)
    if src2 == src:
        return False
    write_file(path, src2)
    return True

changed = False

main_py = root / "app" / "main.py"
tts_py = root / "app" / "tts_service.py"

if not main_py.exists():
    sys.exit(2)
if not tts_py.exists():
    sys.exit(3)

changed |= patch_main_py(main_py)
changed |= patch_tts_service_py(tts_py)

sys.exit(0 if changed else 0)
