"""Benchmark record format shared by the renderer and text-to-3D harnesses.

Every measurement carries: name, value (None when not measured), unit, status (measured | not_run | failed |
not_applicable), reason, and a group (correctness | performance | quality | workload). Run metadata carries the
source commit, environment versions, device, config hash, seeds and measurement method. Values are never
invented and missing values are never written as zero.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "neural"))
sys.path.insert(0, str(ROOT / "scripts"))

GROUPS = ("correctness", "performance", "quality", "workload")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def m(name, value=None, unit="", status="measured", reason="", group="performance", **extra) -> dict:
    assert group in GROUPS and status in ("measured", "not_run", "failed", "not_applicable")
    if status != "measured":
        value = None
    return {"name": name, "value": value, "unit": unit, "status": status, "reason": reason, "group": group, **extra}


def timing_stats(samples_s: list[float]) -> dict:
    s = sorted(samples_s)
    p95 = s[min(len(s) - 1, round(0.95 * (len(s) - 1)))]
    return {"n": len(s), "median_ms": 1e3 * statistics.median(s), "p95_ms": 1e3 * p95, "min_ms": 1e3 * s[0],
            "max_ms": 1e3 * s[-1], "runs_ms": [round(1e3 * x, 4) for x in samples_s]}


def run_metadata(cfg: dict, method: str) -> dict:
    from _common import provenance
    return {"config_hash": config_hash(cfg), "config": cfg, "method": method, "started_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "environment": provenance()}


def load_protocol(path) -> dict:
    p = Path(path)
    return json.loads((p if p.is_absolute() else ROOT / p).read_text())


def write(result: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    return out
