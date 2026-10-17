#!/usr/bin/env python3
"""M5 / research E1 (budget row E-A1): Z-Image-Turbo image-prior smoke test on ONE GPU.

  python scripts/m5_image_prior_smoke.py --check                     # preflight only: no download, no GPU work
  python scripts/m5_image_prior_smoke.py --i-approve-gpu-cost        # download pinned snapshot (~33 GB) + run

Exit evidence (NEURAL_RENDERING_ROADMAP.md M5): a receipt with the exact snapshot/config, finite images, measured
limits (time per image, peak device memory, load time) and `geometry_generated: false` — a text-to-image model
produces images, not 3D geometry or calibrated cameras (TEXT_TO_3D_RESEARCH_PLAN.md §1).

Pins (TEXT_TO_3D_RESEARCH_PLAN.md [R1]): model Tongyi-MAI/Z-Image-Turbo @ f332072aa78be7aecdf3ee76d5c247082da564a6,
Apache-2.0 per the model card; at most 4 anchor prompts; 9 scheduler steps, CFG 0 (upstream Turbo settings).
Refuses to run without a visible GPU, a Diffusers build that resolves the pipeline, and enough free disk. Loading is
staged and hash-verified (berkelium_neural.text3d.zimage_triposr); runs on 2x T4 (Kaggle, configs/kaggle/m5_zimage_smoke.json).
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
NEED_DISK_GB = 34                # 30.6 GB pinned files (31.7 GiB incl. small files) + margin


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
    """Memory-bounded bring-up (fits 2x 15 GB T4): every pinned file verified (sha256 / git blob sha1) on download;
    the Qwen3 text encoder encodes all prompts and is freed before the 6B transformer loads; the transformer is cast
    to the requested dtype at load (no converted copy on disk). A precision study runs prompt 0 in both float16 and
    bfloat16 (second GPU if present) and records finiteness, contrast, time and the pixel difference, so the dtype
    used downstream is a measured choice, not an assumption."""
    import gc

    import numpy as np
    import torch
    from PIL import Image

    from berkelium_neural.text3d import zimage_triposr as zt
    t0 = time.time()
    rec = zt.fetch_zimage(a.cache)
    receipt["snapshot"] = {"repo": REPO_ID, "revision": SNAPSHOT, "local_dir": str(zt.zimage_dir(a.cache)),
                           "verified_files": rec, "total_gb": round(sum(v["bytes"] for v in rec.values()) / 2**30, 2),
                           "fetch_s": round(time.time() - t0, 1)}
    n_gpu = torch.cuda.device_count()
    te = {"attempts": []}
    for te_dtype, te_dev in (("bfloat16", "cuda:0"), ("float32", "cpu")):   # recorded order; the first that runs wins
        t0 = time.time()
        try:
            emb = zt.encode_prompts(a.prompts[:4], te_dev, a.cache, dtype=te_dtype)
        except Exception as e:  # noqa: BLE001 - a failed configuration is a recorded result
            te["attempts"].append({"dtype": te_dtype, "device": te_dev, "error": f"{type(e).__name__}: {e}"[:600]})
            gc.collect()
            torch.cuda.empty_cache()
            continue
        finite = all(bool(torch.isfinite(x).all()) for x in emb)
        te["attempts"].append({"dtype": te_dtype, "device": te_dev, "seconds": round(time.time() - t0, 1),
                               "finite": finite, "tokens": [int(x.shape[0]) for x in emb]})
        if finite:
            te.update({"chosen": {"dtype": te_dtype, "device": te_dev}})
            break
    receipt["text_encoder"] = te
    if "chosen" not in te:
        raise RuntimeError(f"text encoder failed in every configuration: {te['attempts']}")
    gc.collect()
    torch.cuda.empty_cache()
    study, gens = {}, {}
    for dt, dev in (("float16", "cuda:0"), ("bfloat16", f"cuda:{1 if n_gpu > 1 else 0}")):
        try:
            if n_gpu == 1 and gens:                       # one GPU: never two transformers resident at once
                gens.clear()
                gc.collect()
                torch.cuda.empty_cache()
            g = zt.AnchorGenerator(dev, dt, a.cache)
            img, st = g(emb[0], a.seed, a.steps, a.size)
            study[dt] = {**st, "device": dev, "load_s": g.load_s, "mean": float(np.nan_to_num(img).mean())}
            Image.fromarray((np.clip(np.nan_to_num(img), 0, 1) * 255).round().astype(np.uint8)).save(
                a.out / f"precision_{dt}.png")
            gens[dt] = (g, img)
        except Exception as e:  # noqa: BLE001 - a dtype that fails is a recorded study result
            study[dt] = {"error": f"{type(e).__name__}: {e}"[:600], "device": dev}
    ok = [d for d in ("float16", "bfloat16") if d in gens and study[d]["finite"] and study[d]["std"] > 0.02]
    if "float16" in gens and "bfloat16" in gens:
        study["float16_vs_bfloat16_mean_abs_diff"] = float(np.abs(np.nan_to_num(gens["float16"][1]) -
                                                                 np.nan_to_num(gens["bfloat16"][1])).mean())
    chosen = min(ok, key=lambda d: study[d]["seconds"]) if ok else None
    study["rule"] = "use the fastest dtype whose image is finite and non-blank (std > 0.02)"
    study["chosen"] = chosen
    receipt["precision_study"] = study
    receipt["config"]["dtype"] = chosen
    if chosen is None:
        raise RuntimeError(f"no dtype produced a finite, non-blank image: {study}")
    for d in [d for d in gens if d != chosen]:
        del gens[d]
    gc.collect()
    torch.cuda.empty_cache()
    pipe = gens[chosen][0]
    images = []
    for i, prompt in enumerate(a.prompts[:4]):
        img, st = pipe(emb[i], a.seed + i, a.steps, a.size)
        p = a.out / f"anchor_{i}.png"
        Image.fromarray((np.clip(np.nan_to_num(img), 0, 1) * 255).round().astype(np.uint8)).save(p)
        images.append({"prompt": prompt, "seed": a.seed + i, "file": str(p.relative_to(ROOT)), "size": [a.size, a.size],
                       "sha256": zt.sha256_file(p), "mean": float(np.nan_to_num(img).mean()), **st})
        print(f"anchor {i}: {st['seconds']}s, peak {st['peak_gpu_mib']} MiB, std {st['std']:.3f}", flush=True)
    again, _ = pipe(emb[0], a.seed, a.steps, a.size)
    first = np.asarray(Image.open(a.out / "anchor_0.png"), np.float32) / 255.0
    again8 = np.clip(np.nan_to_num(again), 0, 1)
    receipt["repeatability"] = {"same_prompt_same_seed_max_abs_diff_8bit": float(np.abs(
        (again8 * 255).round() / 255 - first).max()), "note": "same device, same dtype, CPU-seeded initial noise"}
    receipt["images"] = images
    receipt["load_s"] = study[chosen]["load_s"]
    receipt["all_finite_nonconstant"] = all(x["finite"] and x["std"] > 0.02 for x in images)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--i-approve-gpu-cost", action="store_true")
    ap.add_argument("--prompts", nargs="+", default=PROMPTS)
    ap.add_argument("--size", type=int, default=768, help="768: 1024² does not fit a 15 GB T4 (attempt 1 receipt)")
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("BERKELIUM_MODEL_CACHE",
                                                                         ROOT / "data" / "models")))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "m5")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    receipt = {"experiment": "E-A1 (research E1) Z-Image-Turbo image-prior smoke", "milestone": "M5",
               "pins": {"repo": REPO_ID, "revision": SNAPSHOT, "license": "Apache-2.0 (model card; verify LICENSE)"},
               "config": {"size": a.size, "steps": a.steps, "guidance_scale": 0.0, "dtype": None,
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
