#!/usr/bin/env bash
# E-T3 open-world generator on CPU: Shap-E (MIT) + CLIP text tower (MIT), installed into .venv-cpu.
# Downloads ~2.2 GB of Shap-E weights + 0.9 GB CLIP ViT-L/14 into data/models/shap_e (resumable, sha256-verified,
# fails hard on a mismatch), then extracts CLIP's text tower once (no TorchScript; see text3d/shap_e_backend.py).
# Needs ~3.5 GB free disk. Nothing here runs on a GPU; nothing is trained.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PYTHON:-.venv-cpu/bin/python}"     # on the MI300X: PYTHON=<the ROCm env's python>
SHAP_E_PIN=50131012ee11c9d2617f3886c10f000d3c7a3b43
command -v "$PY" >/dev/null || [[ -x "$PY" ]] || { echo "run scripts/setup_cpu.sh first (.venv-cpu) or set PYTHON"; exit 2; }
[[ -d third_party/shap-e/.git ]] || git clone -q https://github.com/openai/shap-e.git third_party/shap-e
git -C third_party/shap-e checkout -q "$SHAP_E_PIN"
# CLIP depends on torchvision; on a CPU torch build install the matching CPU torchvision first, or pip pulls a PyPI
# build whose C++ ops do not register against torch+cpu ("operator torchvision::nms does not exist").
if "$PY" -c 'import sys, torch; sys.exit(0 if torch.__version__.endswith("+cpu") else 1)'; then
  "$PY" -m pip install -q --index-url https://download.pytorch.org/whl/cpu torchvision==0.29.1+cpu
fi
"$PY" -m pip install -q -r configs/environments/openworld-cpu.txt
"$PY" -m pip install -q --no-deps -e third_party/shap-e
D=data/models/shap_e; mkdir -p "$D"
B=https://openaipublic.azureedge.net/main/shap-e
for f in text_cond_config.yaml vector_decoder_config.yaml diffusion_config.yaml text_cond.pt vector_decoder.pt; do
  curl -sS -L -C - --retry 5 --retry-delay 5 -o "$D/$f" "$B/$f" || [[ $? -eq 33 ]]   # 33: already complete
done
curl -sS -L -C - --retry 5 --retry-delay 5 -o "$D/ViT-L-14.pt" \
  https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt \
  || [[ $? -eq 33 ]]
( cd "$D" && sha256sum -c - <<'H'
e6b4fa599a7b3c3b16c222d5f5fe56f9db9289ff0b6575fbe5c11bc97106aad4  text_cond.pt
d7e7ebbfe3780499ae89b2da5e7c1354012dba5a6abfe295bed42f25c3be1b98  vector_decoder.pt
b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836  ViT-L-14.pt
f290beeea3d3e9ff15db01bde5382b6e549e463060c0744f89c049505be246c1  text_cond_config.yaml
e6d373649f8e24d85925f4674b9ac41c57aba5f60e42cde6d10f87381326365c  vector_decoder_config.yaml
efcb2cd7ee545b2d27223979d41857802448143990572a42645cd09c2942ed57  diffusion_config.yaml
H
)
[[ -f "$D/clip_vit_l14_text.pt" ]] || "$PY" -c "
import json, sys; sys.path.insert(0, 'services/neural')
from berkelium_neural.text3d.shap_e_backend import prepare_clip_text
print(json.dumps(prepare_clip_text(), indent=2))"
echo "next: $PY scripts/openworld_generate.py 'a wooden chair with four legs' --steps 16   [--device cuda on a GPU]"
