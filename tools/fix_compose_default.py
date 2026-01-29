import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_path = root / ".env"

candidates = [
    ("docker-compose.yml", ["docker-compose.override.yml", "docker-compose.override.yaml"]),
    ("docker-compose.yaml", ["docker-compose.override.yml", "docker-compose.override.yaml"]),
    ("compose.yml", ["compose.override.yml", "compose.override.yaml"]),
    ("compose.yaml", ["compose.override.yml", "compose.override.yaml"]),
]

required_services = ["voice_bot", "tts_service", "llm_server"]

def run_config(files):
    cmd = ["docker", "compose"]
    for f in files:
        cmd += ["-f", str(root / f)]
    cmd += ["config"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout

def service_has_build_or_image(cfg, name):
    lines = cfg.splitlines()
    in_services = False
    in_target = False
    for line in lines:
        if line.strip() == "services:":
            in_services = True
            continue
        if not in_services:
            continue
        m = re.match(r"^\s{2}([A-Za-z0-9._-]+):\s*$", line)
        if m:
            in_target = (m.group(1) == name)
            continue
        if in_target and re.match(r"^\s{4}(image|build):\s*", line):
            return True
    return False

def pick_files():
    for main, overrides in candidates:
        if not (root / main).exists():
            continue
        files = [main]
        for ov in overrides:
            if (root / ov).exists():
                files.append(ov)
        code, cfg = run_config(files)
        if code != 0:
            continue
        ok = True
        for s in required_services:
            if not service_has_build_or_image(cfg, s):
                ok = False
                break
        if ok:
            return files
    return None

def upsert_env_compose_file(files):
    line = "COMPOSE_FILE=" + ":".join(files)
    if env_path.exists():
        txt = env_path.read_text(encoding="utf-8")
        lines = txt.splitlines()
        out = []
        replaced = False
        for l in lines:
            if l.startswith("COMPOSE_FILE="):
                out.append(line)
                replaced = True
            else:
                out.append(l)
        if not replaced:
            if out and out[-1] != "":
                out.append("")
            out.append(line)
        env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    else:
        env_path.write_text(line + "\n", encoding="utf-8")

files = pick_files()
if not files:
    raise SystemExit(1)

upsert_env_compose_file(files)
print("COMPOSE_FILE=" + ":".join(files))
