#!/usr/bin/env python3
"""M5 / research E1 (budget row E-A1): Z-Image-Turbo image-prior smoke test on ONE GPU.

  python scripts/m5_image_prior_smoke.py --check                     # preflight only: no download, no GPU work
  python scripts/m5_image_prior_smoke.py --i-approve-gpu-cost        # download pinned snapshot (~33 GB) + run

Exit evidence (NEURAL_RENDERING_ROADMAP.md M5): a receipt with the exact snapshot/config, finite images, measured
limits (time per image, peak device memory, load time) and `geometry_generated: false` — a text-to-image model
produces images, not 3D geometry or calibrated cameras (TEXT_TO_3D_RESEARCH_PLAN.md §1).

Pins (TEXT_TO_3D_RESEARCH_PLAN.md [R1]): model Tongyi-MAI/Z-Image-Turbo @ f332072aa78be7aecdf3ee76d5c247082da564a6,
Apache-2.0 per the model card; at most 4 anchor prompts; 9 scheduler steps, CFG 0 (upstream Turbo settings).
Refuses to run without a visible GPU, a Diffusers build that resolves the pipeline, and enough free disk.
Results: results/m5/E-A1_receipt.json + results/m5/anchor_*.png. A failed run writes its receipt with the error.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from _common import ROOT, provenance

REPO_ID = "Tongyi-MAI/Z-Image-Turbo"
SNAPSHOT = "f332072aa78be7aecdf3ee76d5c247082da564a6"
PROMPTS = [  # object-centric anchors for the later TripoSR stage (plain background, whole object in frame)
    "a red ceramic teapot on a plain light grey background, studio product photo, whole object visible",
    "a small wooden toy car on a plain light grey background, studio product photo, whole object visible",
    "a blue sneaker on a plain light grey background, studio product photo, whole object visible",
    "a green potted cactus on a plain light grey background, studio product photo, whole object visible",
]
NEED_DISK_GB = 40


def preflight(cache: Path) -> dict:
    out = {"checks": {}, "blockers": []}
    try:
        import torch
        dev = torch.cuda.is_available()
        out["checks"]["torch"] = torch.__version__
        out["checks"]["torch_hip"] = getattr(torch.version, "hip", None)
        out["checks"]["gpu_visible"] = dev
        if dev:
            p = torch.cuda.get_device_properties(0)
            out["checks"]["gpu"] = {"name": p.name, "total_gb": round(p.total_memory / 2**30, 1)}
        else:
            out["blockers"].append("no GPU visible to PyTorch (a 6B diffusion transformer is not a CPU smoke test)")
    except ImportError:
        out["blockers"].append("torch not installed")
    try:
        import diffusers
        out["checks"]["diffusers"] = diffusers.__version__
        out["checks"]["ZImagePipeline"] = hasattr(diffusers, "ZImagePipeline")
        if not hasattr(diffusers, "ZImagePipeline"):
            out["blockers"].append(f"diffusers {diffusers.__version__} has no ZImagePipeline; install a release that "
                                   "does (configs/environments/zimage-rocm.txt) — do not install floating main")
    except ImportError:
        out["blockers"].append("diffusers not installed (configs/environments/zimage-rocm.txt)")
    try:
        import huggingface_hub
        out["checks"]["huggingface_hub"] = huggingface_hub.__version__
    except ImportError:
        out["blockers"].append("huggingface_hub not installed")
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free / 2**30
    out["checks"]["disk_free_gb"] = round(free, 1)
    if free < NEED_DISK_GB:
        out["blockers"].append(f"{free:.1f} GB free at {cache}; the pinned snapshot needs ~33 GB plus staging "
                               f"(require {NEED_DISK_GB} GB)")
    return out


def run(a, receipt: dict):
    import numpy as np
    import torch
    from diffusers import ZImagePipeline
    from huggingface_hub import snapshot_download
    t0 = time.time()
    local = snapshot_download(REPO_ID, revision=SNAPSHOT, cache_dir=str(a.cache))
    files = sorted(p for p in Path(local).rglob("*") if p.is_file())
    receipt["snapshot"] = {"repo": REPO_ID, "revision": SNAPSHOT, "local_dir": local,
                           "files": [{"path": str(p.relative_to(local)), "bytes": p.stat().st_size} for p in files],
                           "total_gb": round(sum(p.stat().st_size for p in files) / 2**30, 2),
                           "download_s": round(time.time() - t0, 1)}
    lic = Path(local) / "LICENSE"
    receipt["snapshot"]["license_file_present"] = lic.exists()
    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(local, torch_dtype=torch.bfloat16).to("cuda")
    torch.cuda.synchronize()
    receipt["load_s"] = round(time.time() - t0, 1)
    receipt["memory_after_load_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    images = []
    for i, prompt in enumerate(a.prompts[:4]):
        torch.cuda.reset_peak_memory_stats()
        gen = torch.Generator("cuda").manual_seed(a.seed + i)
        torch.cuda.synchronize()
        t0 = time.time()
        img = pipe(prompt=prompt, height=a.size, width=a.size, num_inference_steps=a.steps, guidance_scale=0.0,
                   generator=gen).images[0]
        torch.cuda.synchronize()
        dt = time.time() - t0
        arr = np.asarray(img, dtype=np.float32) / 255.0
        p = a.out / f"anchor_{i}.png"
        img.save(p)
        images.append({"prompt": prompt, "seed": a.seed + i, "file": str(p.relative_to(ROOT)), "seconds": round(dt, 2),
                       "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "size": list(img.size),
                       "finite": bool(np.isfinite(arr).all()), "std": float(arr.std()), "mean": float(arr.mean())})
        print(f"anchor {i}: {dt:.1f}s, peak {images[-1]['peak_gib']} GiB, std {images[-1]['std']:.3f}")
    # repeatability: same prompt + seed again
    gen = torch.Generator("cuda").manual_seed(a.seed)
    again = np.asarray(pipe(prompt=a.prompts[0], height=a.size, width=a.size, num_inference_steps=a.steps,
                            guidance_scale=0.0, generator=gen).images[0], dtype=np.float32) / 255.0
    first = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(a.out / "anchor_0.png"), dtype=np.float32) / 255.0
    receipt["repeatability"] = {"same_prompt_same_seed_max_abs_diff": float(np.abs(again - first).max()),
                                "bitwise_equal": bool((again == first).all())}
    receipt["images"] = images
    receipt["all_finite_nonconstant"] = all(x["finite"] and x["std"] > 0.02 for x in images)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--i-approve-gpu-cost", action="store_true")
    ap.add_argument("--prompts", nargs="+", default=PROMPTS)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("BERKELIUM_MODEL_CACHE",
                                                                         ROOT / "data" / "models")))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "m5")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    receipt = {"experiment": "E-A1 (research E1) Z-Image-Turbo image-prior smoke", "milestone": "M5",
               "pins": {"repo": REPO_ID, "revision": SNAPSHOT, "license": "Apache-2.0 (model card; verify LICENSE)"},
               "config": {"size": a.size, "steps": a.steps, "guidance_scale": 0.0, "dtype": "bfloat16",
                          "prompts": a.prompts[:4], "seed": a.seed},
               "geometry_generated": False,
               "note": "text-to-image output only: no 3D geometry, no calibrated camera, no multi-view consistency",
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "environment": provenance()}
    pf = preflight(a.cache)
    receipt["preflight"] = pf
    if a.check or pf["blockers"] or not a.i_approve_gpu_cost:
        receipt["status"] = "blocked_environment" if pf["blockers"] else ("preflight_ok" if a.check else "not_approved")
        (a.out / "E-A1_preflight.json").write_text(json.dumps(receipt, indent=2, default=str))
        print(json.dumps({"status": receipt["status"], "blockers": pf["blockers"], "checks": pf["checks"]}, indent=2))
        return 0 if a.check or receipt["status"] == "preflight_ok" else 2
    try:
        run(a, receipt)
        receipt["status"] = "completed" if receipt.get("all_finite_nonconstant") else "completed_with_bad_images"
    except Exception as e:  # noqa: BLE001 - failed runs keep their receipt
        receipt["status"] = "failed"
        receipt["error"] = f"{type(e).__name__}: {e}"
        receipt["traceback"] = traceback.format_exc()
    (a.out / "E-A1_receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
    print(f"{receipt['status']}: {a.out / 'E-A1_receipt.json'}")
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
