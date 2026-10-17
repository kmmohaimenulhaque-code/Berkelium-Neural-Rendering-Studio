#!/usr/bin/env python3
"""Place berkelium-cad-v1 from an attached Kaggle kernel output into data/cad_v1 and verify every file against the
pinned hashes in configs/datasets/cad_v1.sha256.json (written when the dataset job was pulled and verified)."""
import glob
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pins = json.loads((ROOT / "configs/datasets/cad_v1.sha256.json").read_text())
src = [Path(p).parent for p in glob.glob("/kaggle/input/**/data/cad_v1/manifest.json", recursive=True)]
if not src:
    sys.exit("berkelium-cad-v1 not attached (no */data/cad_v1/manifest.json under /kaggle/input)")
dst = ROOT / "data" / "cad_v1"
dst.mkdir(parents=True, exist_ok=True)
bad = []
for name, h in pins.items():
    shutil.copy2(src[0] / name, dst / name)
    if hashlib.sha256((dst / name).read_bytes()).hexdigest() != h:
        bad.append(name)
print(json.dumps({"source": str(src[0]), "files": len(pins), "bad": bad}))
sys.exit(1 if bad else 0)
