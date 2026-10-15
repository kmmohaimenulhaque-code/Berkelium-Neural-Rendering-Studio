"""One Shap-E generation in its own process: ``python -m berkelium_neural.text3d.shap_e_worker --prompt ... --out D``.

Used by the service (text_to_3d jobs with generator shap-e-text300m / auto) and by scripts/openworld_generate.py, so a
heavy generation never shares a heap with the API server and a memory-watchdog kill cannot take the service down.
Writes D/mesh.npz (exact float32 arrays) and D/report.json; prints one JSON object per line on stdout:
  {"event": "progress", "frac": f, "stage": s, "rss_mib": {...}}   per stage / denoiser call
  {"event": "done", "status": "ok" | "invalid_mesh", "report": {...}}
The process aborts itself (exit 86, report.json status "aborted_memory") when VmRSS exceeds --rss-abort-mib: this
host's watchdog SIGTERMs processes at ~3.2 GB RSS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ABORT_EXIT = 86
_KEYS = ("VmRSS", "RssAnon", "RssFile")


def rss_mib() -> dict:
    out = {}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                k = line.split(":")[0]
                if k in _KEYS:
                    out[k] = int(line.split()[1]) // 1024
    except OSError:
        pass
    return out


class Monitor:
    """Samples /proc/self/status every 50 ms; tracks peaks; aborts the process above ``limit_mib``."""

    def __init__(self, limit_mib: int, report_path: Path):
        self.limit, self.report_path = limit_mib, report_path
        self.peak = dict.fromkeys(_KEYS, 0)
        self.stage = "start"
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            st = rss_mib()
            for k, v in st.items():
                self.peak[k] = max(self.peak[k], v)
            if st.get("VmRSS", 0) > self.limit:
                self.report_path.parent.mkdir(parents=True, exist_ok=True)
                self.report_path.write_text(json.dumps({"status": "aborted_memory", "stage": self.stage,
                                                        "rss_mib": st, "limit_mib": self.limit}, indent=2))
                print(json.dumps({"event": "abort", "reason": "memory", "rss_mib": st, "stage": self.stage}),
                      flush=True)
                os._exit(ABORT_EXIT)
            time.sleep(0.05)


def run(prompt: str, out: Path, seed: int = 0, steps: int = 16, guidance: float = 15.0, threads: int = 2,
        grid_size: int | None = None, clip_dtype: str = "float32", rss_abort_mib: int = 3100,
        emit=print) -> dict:
    from .shap_e_backend import ShapEGenerator
    out = Path(out)
    mon = Monitor(rss_abort_mib, out / "report.json")
    gen = ShapEGenerator(steps=steps, guidance_scale=guidance, threads=threads, grid_size=grid_size,
                         clip_dtype=clip_dtype)
    ok, why = gen.available()
    if not ok:
        raise RuntimeError(f"MODEL_UNAVAILABLE: {why}")
    last = {"stage": None, "t": 0.0}

    def progress(frac, stage):
        mon.stage = stage
        now = time.time()
        if stage != last["stage"] or now - last["t"] > 2.0:
            last.update(stage=stage, t=now)
            emit(json.dumps({"event": "progress", "frac": round(float(frac), 4), "stage": stage, "rss_mib": rss_mib()}))
    t0 = time.time()
    o = gen.generate(prompt, seed, progress=progress)
    rep = o["report"]
    out.mkdir(parents=True, exist_ok=True)
    if o["vertices"] is not None:
        np.savez_compressed(out / "mesh.npz", v=o["vertices"], f=o["faces"],
                            col=o["colors"] if o["colors"] is not None else np.zeros((0, 3), np.float32))
    rep.update({"status": "ok" if rep.get("valid") else "invalid_mesh", "identity": gen.identity, "threads": threads,
                "wall_s": round(time.time() - t0, 1), "seconds_per_step": round(rep["sample_s"] / steps, 2),
                "memory_mib": {"peak_" + k: v for k, v in mon.peak.items()}, "rss_abort_mib": rss_abort_mib})
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    emit(json.dumps({"event": "done", "status": rep["status"], "report": rep}, default=str))
    return {**o, "report": rep}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=15.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--grid-size", type=int, default=None)
    ap.add_argument("--clip-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--rss-abort-mib", type=int, default=3100)
    a = ap.parse_args(argv)
    o = run(a.prompt, Path(a.out), a.seed, a.steps, a.guidance, a.threads, a.grid_size, a.clip_dtype,
            a.rss_abort_mib, emit=lambda s: print(s, flush=True))
    return 0 if o["report"].get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())
