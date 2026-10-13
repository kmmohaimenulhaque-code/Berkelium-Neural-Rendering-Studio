"""Fixed evaluation protocol for text-to-3D on berkelium-procedural-shapes/1.

For every evaluation record: generate from the caption, then compare against the INDEPENDENT analytic ground truth:
  geometry   Chamfer-L1/L2, normal consistency, F-score@0.02/0.05 (20k area-weighted samples per mesh)
  validity   non-empty, finite, in bounds, degenerate faces, watertight, edge-manifold, components
  parts      predicted connected-component count == component count of the GT mesh (touching parts merge; a
             cone stacked on a cone apex touches at a point and stays 2 components)
  colour     per-part mean predicted colour -> nearest named colour == requested colour
  views      PSNR / SSIM of the predicted mesh vs stored GT renders, same cameras, same z-buffer renderer, 8-bit
  LPIPS      not_run (requires downloading pretrained network weights; not approved)
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

import numpy as np

from ..contracts import DatasetRecord
from .dataset import composition_of, load_manifest
from .geometry import geometry_metrics, grid_points, mesh_from_sdf_grid, psnr, render_mesh, ssim, validate_mesh
from .shapes import COLORS, colors_at, sdf

PROTOCOL = "berkelium-text3d-eval/1"


class RetrievalBaseline:
    """Non-learned baseline: returns the ground-truth shape of the training caption with the highest token Jaccard
    overlap (ties -> lowest asset id). Real geometry, but cannot compose unseen structures."""

    name = "retrieval_baseline"

    def __init__(self, train_records: list[DatasetRecord], resolution: int = 48):
        self.items = sorted(((set(re.findall(r"[a-z]+", r.caption.lower())), r) for r in train_records),
                            key=lambda x: x[1].asset_id)
        self.res = resolution

    def generate(self, prompt: str):
        t0 = time.time()
        q = set(re.findall(r"[a-z]+", prompt.lower()))
        best = max(self.items, key=lambda x: len(q & x[0]) / max(1, len(q | x[0])))[1]
        c = composition_of(best)
        vol = sdf(c, grid_points(self.res)).reshape((self.res,) * 3)
        v, f = mesh_from_sdf_grid(vol)
        rep = validate_mesh(v, f)
        rep.update({"retrieved": best.asset_id, "seconds": round(time.time() - t0, 3)})
        return {"vertices": v, "faces": f, "colors": None if v is None else colors_at(c, v), "report": rep}


class ModelGenerator:
    def __init__(self, model, resolution: int = 48, name: str = "model"):
        self.model, self.res, self.name = model, resolution, name

    def generate(self, prompt: str):
        from .generate import generate_mesh
        return generate_mesh(self.model, prompt, self.res)


def nearest_color(rgb) -> str:
    return min(COLORS, key=lambda k: float(np.sum((np.array(COLORS[k]) - rgb) ** 2)))


def evaluate_record(gen, r: DatasetRecord, root: Path, n_samples=20000) -> dict:
    from PIL import Image
    out = gen.generate(r.caption)
    v, f, col = out["vertices"], out["faces"], out["colors"]
    gt = np.load(root / r.files["gt_mesh"])
    c = composition_of(r)
    row = {"asset_id": r.asset_id, "family": r.family, "caption": r.caption, "generator": gen.name,
           "seconds": out["report"]["seconds"]}
    rep = out["report"]
    row["success"] = bool(rep.get("valid", False))
    row.update({k: rep.get(k) for k in ("n_faces", "components", "watertight", "edge_manifold", "degenerate_faces",
                                        "in_bounds")})
    row.update(geometry_metrics(v, f, gt["v"], gt["f"], n=n_samples))
    row["gt_components"] = int(gt["components"])
    row["part_count_correct"] = bool(rep.get("components") == row["gt_components"]) if row["success"] else False
    if row["success"] and col is not None:
        _, part = sdf(c, v, return_part=True)
        ok, err = [], []
        for i, p in enumerate(c.parts):
            sel = part == i
            if not sel.any():
                ok.append(False)
                continue
            m = col[sel].mean(0)
            ok.append(nearest_color(m) == p.color)
            err.append(float(np.abs(m - np.array(COLORS[p.color])).mean()))
        row["color_parts_correct"], row["color_parts_total"] = int(sum(ok)), len(ok)
        row["color_mae"] = float(np.mean(err)) if err else None
    else:
        row["color_parts_correct"], row["color_parts_total"], row["color_mae"] = 0, len(c.parts), None
    ps, ss = [], []
    for vi, cam in enumerate(r.cameras):
        ref = np.asarray(Image.open(root / r.files[f"view{vi}"]), dtype=np.float32) / 255.0
        img, _, _ = render_mesh(v, f, col if col is not None else np.full((len(v) if v is not None else 0, 3), 0.7),
                                cam)
        img = np.round(np.clip(img, 0, 1) * 255) / 255.0
        ps.append(psnr(img, ref))
        ss.append(ssim(img.astype(np.float32), ref))
    row["view_psnr"] = float(np.mean(ps)) if ps else None
    row["view_ssim"] = float(np.mean(ss)) if ss else None
    row["lpips"] = None
    return row


def _stats(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return {"n": 0, "mean": None, "std": None, "median": None}
    return {"n": len(xs), "mean": float(np.mean(xs)), "std": float(np.std(xs)), "median": float(statistics.median(xs))}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    agg = {"n_records": n, "success_rate": sum(r["success"] for r in rows) / n if n else None,
           "part_count_accuracy": sum(r["part_count_correct"] for r in rows) / n if n else None,
           "color_accuracy": (sum(r["color_parts_correct"] for r in rows) / max(1, sum(r["color_parts_total"]
                              for r in rows))) if n else None}
    for k in ("chamfer_l1", "chamfer_l2", "normal_consistency", "fscore@0.02", "fscore@0.05", "view_psnr",
              "view_ssim", "color_mae", "seconds"):
        agg[k] = _stats([r.get(k) for r in rows])
    agg["lpips"] = {"status": "not_run", "reason": "requires pretrained LPIPS network weights (download not approved)"}
    return agg


def evaluate(gen, root: Path, split: str, limit: int | None = None, log=print) -> dict:
    root = Path(root)
    recs = load_manifest(root, {split})[: limit or None]
    rows = []
    for i, r in enumerate(recs):
        rows.append(evaluate_record(gen, r, root))
        if log and (i + 1) % 10 == 0:
            log(f"  [{gen.name}|{split}] {i + 1}/{len(recs)}")
    card = json.loads((root / "dataset_card.json").read_text())
    return {"protocol": PROTOCOL, "generator": gen.name, "split": split, "split_hash": card["split_hashes"].get(split),
            "dataset_manifest_sha256": card["manifest_sha256"], "aggregate": aggregate(rows), "records": rows}
