"""Berkelium benchmarks. Importing this package makes services/neural (berkelium_neural) and scripts/ importable
without installation, before any submodule's own imports run."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "services" / "neural", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
