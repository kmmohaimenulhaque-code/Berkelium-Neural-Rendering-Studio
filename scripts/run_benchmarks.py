#!/usr/bin/env python3
"""Run ONE approved experiment and store its full, machine-readable result.

  python scripts/run_benchmarks.py --experiment configs/experiments/E-R0_renderer_cpu.json
  python scripts/run_benchmarks.py --experiment configs/experiments/E-R1_renderer_mi300x.json --i-approve-gpu-cost

An experiment whose config has "approved": false (GPU credit) is refused unless --i-approve-gpu-cost is passed AND its
"requires" are actually satisfied (ROCm PyTorch, visible device, gsplat import). Results are written to the configured
output with a timestamped copy; failed runs are preserved with their error, never discarded. The result stores the
experiment config verbatim (its static "approved" value is unchanged) plus a separate "authorization" record saying
whether this run was authorized by the config or by the per-run --i-approve-gpu-cost flag.
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


def run_authorization(exp: dict, cli_flag: bool) -> dict:
    """Runtime authorization record. The experiment config (``exp``, stored verbatim in the result) keeps its static
    ``approved`` value; a per-run operator authorization given with --i-approve-gpu-cost is recorded separately and
    is never written back as ``approved: true``."""
    static = bool(exp.get("approved"))
    by = "static_config" if static else ("cli_flag" if cli_flag else None)
    return {"static_config_approved": static, "cli_i_approve_gpu_cost": bool(cli_flag), "authorized_by": by,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": ("per-run authorization asserted by the operator at invocation (--i-approve-gpu-cost); the "
                     "experiment config was not modified") if by == "cli_flag" else ""}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--i-approve-gpu-cost", action="store_true")
    a = ap.parse_args(argv)
    exp = load_json(a.experiment)
    auth = run_authorization(exp, a.i_approve_gpu_cost)
    if auth["authorized_by"] is None:
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
        res = {"experiment": exp, "authorization": auth, "status": "completed", **run(proto)}
    except Exception as e:  # noqa: BLE001
        res = {"experiment": exp, "authorization": auth, "status": "failed", "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()}
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = out.with_name(f"{out.stem}.{stamp}{out.suffix}")
    hist.write_text(json.dumps(res, indent=2, default=str))
    shutil.copy(hist, out)
    print(f"{res['status']}: {out} (history: {hist.name})")
    return 0 if res["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
