#!/usr/bin/env bash
# Start the Berkelium backend (chat/OCR/providers + /v1/neural text-to-3D, rendering, durable jobs) on :8000.
#   bash scripts/start_backend.sh            # uses .venv-cpu (scripts/setup_cpu.sh --install) or $PYTHON
# Jobs and artifacts persist in $BERKELIUM_NEURAL_HOME (default ~/.berkelium/neural) and survive restarts.
# config.json (project root) is optional: without it chat is disabled and no API key is enforced.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv-cpu/bin/python}"
[[ -x "$PY" ]] || { echo "no Python at $PY; run: bash scripts/setup_cpu.sh --install (or set PYTHON=...)"; exit 2; }
export BERKELIUM_NEURAL_HOME="${BERKELIUM_NEURAL_HOME:-$HOME/.berkelium/neural}"
cd "$ROOT/services/backend"
echo "backend: http://127.0.0.1:${PORT:-8000}  (neural home: $BERKELIUM_NEURAL_HOME)"
exec "$PY" -m uvicorn main:app --host 127.0.0.1 --port "${PORT:-8000}"
