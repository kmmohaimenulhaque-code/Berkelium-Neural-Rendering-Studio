#!/usr/bin/env bash
# Reproducible CPU reference environment. Default: CHECK + PLAN only. Pass --install to act.
#   scripts/setup_cpu.sh            # validate prerequisites, print what would be done
#   scripts/setup_cpu.sh --install  # create .venv-cpu, install pinned deps, fetch pinned third_party sources
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
INSTALL=0; [[ "${1:-}" == "--install" ]] && INSTALL=1
PY="${PYTHON:-python3}"; VENV="${VENV:-.venv-cpu}"
GSPLAT_REPO=https://github.com/AMD-Ecosystem/gsplat.git; GSPLAT_COMMIT=ff8bc5489545bcc1b5a2b997d975834429a0d7af
HIPCPU_REPO=https://github.com/ROCm/HIP-CPU.git;        HIPCPU_COMMIT=e112c935057434897bb12d9ab3910380a8bd5f58
fail=0
need() { if ! command -v "$1" >/dev/null; then echo "MISSING: $1 ($2)"; fail=1; else echo "ok: $1"; fi; }
need "$PY" "Python >= 3.10"; need git "clone pinned sources"; need g++ "C++17 compiler for HIP-CPU"; need node "Studio app (optional)"
"$PY" -c 'import sys; assert sys.version_info >= (3,10), sys.version' || { echo "MISSING: Python >= 3.10"; fail=1; }
if [[ -f /usr/include/tbb/tbb.h || -f /usr/include/oneapi/tbb.h ]]; then echo "ok: TBB headers"; else
  echo "MISSING: TBB (Ubuntu: sudo apt-get install libtbb-dev) — needed only for the HIP-CPU kernel tests"; fi
free_gb=$(df -Pk "$ROOT" | awk 'NR==2{printf "%d", $4/1048576}'); echo "disk free: ${free_gb} GB (need ~3 GB)"
(( free_gb >= 3 )) || { echo "MISSING: disk space"; fail=1; }
echo; echo "PLAN:"
echo "  1. $PY -m venv $VENV && pip install torch==2.14.1 (CPU wheel) + configs/environments/cpu.txt"
echo "  2. git clone $GSPLAT_REPO third_party/amd-gsplat @ $GSPLAT_COMMIT  (source only; NOT built on CPU)"
echo "  3. git clone $HIPCPU_REPO third_party/hip-cpu @ $HIPCPU_COMMIT  (header-only runtime)"
echo "  4. (cd apps/desktop && ELECTRON_SKIP_BINARY_DOWNLOAD=1 npm ci); python -m playwright install chromium"
(( fail == 0 )) || { echo "prerequisites missing — nothing installed"; exit 2; }
(( INSTALL == 1 )) || { echo "check only. Re-run with --install to perform the plan."; exit 0; }
"$PY" -m venv "$VENV"; source "$VENV/bin/activate"
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r configs/environments/cpu.txt
clone() { [[ -d "$2/.git" ]] || git clone -q "$1" "$2"; git -C "$2" fetch -q --all --tags; git -C "$2" checkout -q "$3";
          echo "pinned $2 @ $(git -C "$2" rev-parse HEAD)"; }
mkdir -p third_party; clone "$GSPLAT_REPO" third_party/amd-gsplat "$GSPLAT_COMMIT"; clone "$HIPCPU_REPO" third_party/hip-cpu "$HIPCPU_COMMIT"
python scripts/doctor.py
echo "next: python -m pytest services/neural/tests -q"
