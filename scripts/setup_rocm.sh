#!/usr/bin/env bash
# AMD ROCm environment for the gsplat baseline + Berkelium HIP kernel (MI300X / gfx942).
# Default: CHECK + PLAN only (no installs, no GPU allocation). --install performs the plan; --smoke runs ONE tiny
# GPU smoke test (allocates the GPU briefly). Never installs a second PyTorch over an existing ROCm PyTorch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
INSTALL=0; SMOKE=0; for a in "$@"; do [[ $a == --install ]] && INSTALL=1; [[ $a == --smoke ]] && SMOKE=1; done
PY="${PYTHON:-python3}"
GSPLAT_COMMIT=ff8bc5489545bcc1b5a2b997d975834429a0d7af
fail=0
ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo none); echo "ROCm: $ROCM_VER"
[[ $ROCM_VER == none ]] && { echo "MISSING: ROCm (/opt/rocm). Use rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0"; fail=1; }
command -v hipcc >/dev/null && echo "ok: hipcc" || { echo "MISSING: hipcc"; fail=1; }
GFX=$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | sort -u | tr '\n' ' ' || true); echo "GPU arch: ${GFX:-none}"
[[ $GFX == *gfx942* ]] || echo "WARNING: gfx942 (MI300X) not detected; AMD gsplat is only supported on MI300X"
TORCH=$("$PY" -c 'import torch,sys;print(torch.__version__, torch.version.hip)' 2>/dev/null || echo "none none")
echo "torch: $TORCH"
case "$TORCH" in
  none*) echo "MISSING: ROCm PyTorch. Use the AMD image (configs/environments/rocm-*.txt); this script will not pip-install torch."; fail=1;;
  *" None") echo "INVALID: this torch is not a ROCm build (torch.version.hip is None). Do not mix CUDA and ROCm."; fail=1;;
esac
PYV=$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
TV=$(echo "$TORCH" | cut -d. -f1-2)
case "$PYV/$TV" in 3.12/2.8|3.10/2.6) echo "ok: Python $PYV + torch $TV is in the AMD gsplat support matrix";;
  *) echo "WARNING: Python $PYV + torch $TV is NOT in the AMD support matrix (3.12+2.8 / 3.10+2.6): build from source and run tests";; esac
echo; echo "PLAN:"
echo "  1. pip install -r configs/environments/rocm-7.0-pt2.8-py3.12.txt   (no torch)"
echo "  2. gsplat from pinned source: third_party/amd-gsplat @ $GSPLAT_COMMIT -> pip install --no-build-isolation ."
echo "     (alternative: pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-7.0.0/simple/)"
echo "  3. python scripts/doctor.py ; Berkelium kernel builds on first use (hipcc via torch.utils.cpp_extension)"
echo "  4. optional --smoke: one 256x256 render + backward with gsplat and with the Berkelium kernel"
(( fail == 0 )) || { echo "prerequisites missing — nothing installed"; exit 2; }
(( INSTALL == 1 )) || { echo "check only. Re-run with --install (and optionally --smoke)."; exit 0; }
pip install -r configs/environments/rocm-7.0-pt2.8-py3.12.txt
[[ -d third_party/amd-gsplat/.git ]] || git clone -q https://github.com/AMD-Ecosystem/gsplat.git third_party/amd-gsplat
git -C third_party/amd-gsplat checkout -q "$GSPLAT_COMMIT"; git -C third_party/amd-gsplat submodule update --init --recursive
(cd third_party/amd-gsplat && pip install --no-build-isolation .)
"$PY" scripts/doctor.py
if (( SMOKE == 1 )); then
  "$PY" -m pytest services/neural/tests -q -m "hip or gpu" -p no:cacheprovider || true
  "$PY" - <<'PYS'
import sys; sys.path.insert(0, "services/neural")
import torch
from berkelium_neural import camera
from berkelium_neural.contracts import RenderConfig
from berkelium_neural.render.gsplat_adapter import GsplatRenderer
from berkelium_neural.render.interface import GaussianTensors
sys.path.insert(0, "."); from benchmarks.renderer.bench_renderer import synthetic_scene
s = synthetic_scene(2000, 0); cam = camera.make_camera((0, 0, -0.6), (0, 0, 3), 256, 256, fov_y_deg=55)
for name in ("gsplat", "berkelium"):
    g = GaussianTensors.from_scene(s, device="cuda", requires_grad=True)
    o = GsplatRenderer(name).render(g, cam, RenderConfig()); o.rgb.sum().backward(); torch.cuda.synchronize()
    print(name, "ok", o.meta["n_pairs"], float(g.means.grad.abs().sum()))
PYS
fi
