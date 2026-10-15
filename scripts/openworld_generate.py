#!/usr/bin/env python3
"""One open-world text-to-3D generation (Shap-E text300M on CPU) in its own process, with a resource record.

  python scripts/openworld_generate.py "a wooden chair with four legs" --steps 16 --out results/text3d/E-T3/p2

Writes <out>/<slug>__s<seed>/{mesh.glb, mesh.npz (exact float32 arrays), preview.png, report.json}. The report has
stage timings, the generator identity (commit, weight sha256, sampler), mesh validity, and memory sampled from /proc
every 50 ms (VmRSS, RssAnon, RssFile). Generation itself is berkelium_neural.text3d.shap_e_worker.run, the same code
the service runs; the process aborts itself (exit 86, reason recorded) above --rss-abort-mib so a run never reaches
this host's memory watchdog, which SIGTERMs processes at ~3.2 GB RSS.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from _common import ROOT, provenance


def slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:60]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--grid-size", type=int, default=None, help="STF SDF grid (checkpoint default if omitted)")
    ap.add_argument("--clip-dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--out", default=str(ROOT / "results" / "text3d" / "E-T3" / "p2"))
    ap.add_argument("--dir-name", default=None, help="output subdirectory (default <slug>__s<seed>)")
    ap.add_argument("--rss-abort-mib", type=int, default=3100)
    ap.add_argument("--device", default="cpu", help="cpu (verified) or cuda (ROCm/HIP or CUDA; UNVERIFIED)")
    ap.add_argument("--fp32", action="store_true", help="fp32 sampling on a GPU (CPU is always fp32)")
    a = ap.parse_args(argv)
    out = Path(a.out if os.path.isabs(a.out) else ROOT / a.out)
    d = out / (a.dir_name or f"{slug(a.prompt)}__s{a.seed}")

    import numpy as np

    from berkelium_neural.camera import orbit_cameras
    from berkelium_neural.fitting.real_object import contact_sheet, save_png
    from berkelium_neural.text3d.generate import mesh_to_glb
    from berkelium_neural.text3d.geometry import render_mesh
    from berkelium_neural.text3d.shap_e_backend import available
    from berkelium_neural.text3d.shap_e_worker import run
    ok, why = available()
    if not ok:
        print(f"BLOCKED: {why}")
        return 2
    t0 = time.time()
    o = run(a.prompt, d, a.seed, a.steps, a.guidance, a.threads, a.grid_size, a.clip_dtype, a.rss_abort_mib,
            emit=lambda s: print(time.strftime("%T"), s, flush=True), device=a.device,
            fp16=False if a.fp32 else None)
    rep = o["report"]
    if o["vertices"] is not None:
        (d / "mesh.glb").write_bytes(mesh_to_glb(o["vertices"], o["faces"], o["colors"]))
        col = o["colors"] if o["colors"] is not None else np.full((len(o["vertices"]), 3), 0.7, np.float32)
        save_png(d / "preview.png", contact_sheet([[render_mesh(o["vertices"], o["faces"], col, c)[0]
                                                    for c in orbit_cameras(4, 3.2, 25.0, 192, 192)]]))
    rep.update({"script_wall_s": round(time.time() - t0, 1), "environment": provenance(),
                "outputs": {k: str((d / k).relative_to(ROOT)) for k in ("mesh.glb", "mesh.npz", "preview.png")
                            if (d / k).exists()}})
    (d / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep.get(k) for k in ("status", "valid", "n_faces", "components", "sample_s", "decode_s",
                                               "seconds", "seconds_per_step", "memory_mib")}, indent=2))
    return 0 if rep.get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())
