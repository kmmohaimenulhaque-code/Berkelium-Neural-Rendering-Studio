"""Shared helpers for Berkelium scripts: import path, JSON config loading, provenance stamp."""
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "neural"))


def load_json(p) -> dict:
    return json.loads(Path(p).read_text())


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True,
                               check=False).stdout.strip()
        return (r.stdout.strip() + ("+dirty" if dirty else "")) if r.returncode == 0 else None
    except OSError:
        return None


def provenance() -> dict:
    import numpy
    d = {"git_commit": git_commit(), "python": platform.python_version(), "platform": platform.platform(),
         "numpy": numpy.__version__}
    try:
        import torch
        d.update({"torch": torch.__version__, "torch_hip": getattr(torch.version, "hip", None),
                  "torch_cuda": getattr(torch.version, "cuda", None), "gpu_visible": torch.cuda.is_available()})
        if torch.cuda.is_available():
            d["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        d["torch"] = None
    return d
