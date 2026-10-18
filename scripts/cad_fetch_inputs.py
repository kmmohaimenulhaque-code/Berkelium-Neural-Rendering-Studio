#!/usr/bin/env python3
"""Place the outputs of earlier E-C1 kernels (attached as Kaggle kernel sources) into the work tree:
the trained adapter + its run manifest. Every file is checked against
the producing kernel's own manifest.json (sha256), so a stale or partial output is refused."""
import glob
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WANT = {"exports/qwen3-8b-cad-v1/": "adapter"}
placed, bad = {}, []
for man_path in glob.glob("/kaggle/input/**/out/manifest.json", recursive=True):
    base = Path(man_path).parent
    man = json.loads(Path(man_path).read_text())
    for rel, m in man.items():
        if not any(rel.startswith(p) for p in WANT):
            continue
        src = base / "files" / rel
        if hashlib.sha256(src.read_bytes()).hexdigest() != m["sha256"]:
            bad.append(rel)
            continue
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        placed[rel] = m["sha256"][:16]
print(json.dumps({"placed": placed, "bad": bad}, indent=1))
need = [p for p in ("exports/qwen3-8b-cad-v1/adapter/adapter_model.safetensors",) if not (ROOT / p).exists()]
sys.exit(1 if bad or need else 0)
