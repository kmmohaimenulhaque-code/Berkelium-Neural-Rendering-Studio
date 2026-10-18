import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "cadquery", ROOT / "services" / "neural"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
