#!/usr/bin/env bash
# M6 environment: .venv-triposr + pinned TripoSR code (third_party/TripoSR) + pinned weights (data/models).
# CPU only; torchmcubes and rembg are NOT installed (scripts/m6_triposr_controls.py supplies a CPU marching-cubes
# shim and never removes backgrounds; see decision D-020).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-python3}"
CODE_PIN=107cefdc244c39106fa830359024f6a2f1c78871
WEIGHTS_PIN=5b521936b01fbe1890f6f9baed0254ab6351c04a
"$PY" -m venv .venv-triposr
.venv-triposr/bin/pip install -q --upgrade pip
.venv-triposr/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
.venv-triposr/bin/pip install -q -r configs/environments/triposr-cpu.txt
[[ -d third_party/TripoSR/.git ]] || git clone -q https://github.com/VAST-AI-Research/TripoSR.git third_party/TripoSR
git -C third_party/TripoSR checkout -q "$CODE_PIN"
.venv-triposr/bin/python - <<PYEOF
from huggingface_hub import hf_hub_download
for f in ("config.yaml", "model.ckpt"):
    print(hf_hub_download("stabilityai/TripoSR", f, revision="$WEIGHTS_PIN", cache_dir="data/models"))
PYEOF
echo "next: .venv-triposr/bin/python scripts/m6_triposr_controls.py   (needs data/procedural_shapes_v1, test split)"
