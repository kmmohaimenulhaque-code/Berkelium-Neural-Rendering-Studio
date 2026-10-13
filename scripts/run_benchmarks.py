#!/usr/bin/env python3
"""Run ONE approved experiment and store its full, machine-readable result.

  python scripts/run_benchmarks.py --experiment configs/experiments/E-R0_renderer_cpu.json
  python scripts/run_benchmarks.py --experiment configs/experiments/E-R1_renderer_mi300x.json --i-approve-gpu-cost

An experiment whose config has "approved": false (GPU credit) is refused unless --i-approve-gpu-cost is passed AND its
"requires" are actually satisfied (ROCm PyTorch, visible device, gsplat import). Results are written to the configured
output with a timestamped copy; failed runs are preserved with their error, never discarded.
"""
import argparse
import json
import shutil
import sys
import time
import traceback

from _common import ROOT, load_json

sys.path.insert(0, str(ROOT))


def requirements_met(reqs) -> list[str]:
    missing = []
    if "rocm" in reqs:
        try:
            import torch
            if not getattr(torch.version, "hip", None):
                missing.append("rocm: torch.version.hip is None (CUDA/CPU build)")
            elif not torch.cuda.is_available():
                missing.append("rocm: no visible HIP device")
        except ImportError:
            missing.append("rocm: torch not installed")
    if "gsplat" in reqs:
        try:
            import gsplat  # noqa: F401
        except Exception as e:  # noqa: BLE001
            missing.append(f"gsplat: {e}")
    return missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--i-approve-gpu-cost", action="store_true")
    a = ap.parse_args(argv)
    exp = load_json(a.experiment)
    if not exp.get("approved") and not a.i_approve_gpu_cost:
        print(f"refused: {exp['id']} is not approved (see approval_note). Nothing was run.")
        return 3
    missing = requirements_met(exp.get("requires", []))
    if missing:
        print(f"refused: requirements not met for {exp['id']}: {missing}. Nothing was run.")
        return 4
    proto = load_json(ROOT / exp["protocol"])
    out = ROOT / exp["output"]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    try:
        if exp["benchmark"] == "renderer":
            from benchmarks.renderer.bench_renderer import run
        elif exp["benchmark"] == "text3d":
            from benchmarks.text3d.bench_text3d import run
        else:
            raise ValueError(exp["benchmark"])
        res = {"experiment": exp, "status": "completed", **run(proto)}
    except Exception as e:  # noqa: BLE001
        res = {"experiment": exp, "status": "failed", "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()}
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = out.with_name(f"{out.stem}.{stamp}{out.suffix}")
    hist.write_text(json.dumps(res, indent=2, default=str))
    shutil.copy(hist, out)
    print(f"{res['status']}: {out} (history: {hist.name})")
    return 0 if res["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
