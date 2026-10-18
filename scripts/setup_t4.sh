#!/usr/bin/env bash
# Kaggle / Colab T4 (CUDA) experimental environment. CHECK + PLAN by default; --install acts.
# Evidence from this environment is CUDA-only: it does NOT validate the AMD HIP port or the Berkelium HIP kernel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
INSTALL=0; [[ "${1:-}" == "--install" ]] && INSTALL=1
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no NVIDIA GPU visible"
python3 -c 'import torch;print("torch", torch.__version__, "cuda", torch.version.cuda, "hip", torch.version.hip, "gpu", torch.cuda.is_available())'
python3 -c 'import torch,sys; sys.exit(0 if (torch.version.hip is None) else 3)' || { echo "INVALID: ROCm torch in a CUDA environment"; exit 2; }
echo "PLAN: pip install -r configs/environments/t4-cuda.txt (upstream gsplat CUDA, JIT on first use); run CPU tests; text-to-3D training"
(( INSTALL == 1 )) || { echo "check only. Re-run with --install."; exit 0; }
pip install -r configs/environments/t4-cuda.txt
python3 scripts/doctor.py
