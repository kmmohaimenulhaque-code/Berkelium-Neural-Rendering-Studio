"""Text-to-3D benchmark: runs the fixed evaluation protocol (berkelium-text3d-eval/1) for each configured
baseline/candidate pair with identical prompts (the split's captions), resolution and seeds."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

from benchmarks.common import ROOT, m, run_metadata

METRICS = [  # (aggregate key, unit, higher_is_better)
    ("success_rate", "", True), ("part_count_accuracy", "", True), ("color_accuracy", "", True),
    ("chamfer_l1", "normalized_object", False), ("chamfer_l2", "normalized_object^2", False),
    ("normal_consistency", "", True), ("fscore@0.02", "", True), ("fscore@0.05", "", True),
    ("view_psnr", "dB", True), ("view_ssim", "", True), ("color_mae", "", False), ("seconds", "s/prompt", False)]


def _generator(spec: dict, root: Path, res: int):
    from berkelium_neural.text3d.dataset import load_manifest
    from berkelium_neural.text3d.evaluate import ModelGenerator, RetrievalBaseline
    from berkelium_neural.text3d.train import file_sha256, load_model
    if spec["kind"] == "retrieval":
        g = RetrievalBaseline(load_manifest(root, set(spec["train_splits"])), res)
        g.name = "retrieval[" + "+".join(spec["train_splits"]) + "]"
        return g, {"kind": "retrieval", "train_splits": spec["train_splits"]}
    ck = ROOT / spec["checkpoint"]
    ad = ROOT / spec["adapter"] if spec.get("adapter") else None
    if not ck.exists() or (ad is not None and not ad.exists()):
        raise FileNotFoundError(f"checkpoint missing: {ck if not ck.exists() else ad}")
    model = load_model(ck, ad)
    ident = {"kind": "model", "checkpoint": spec["checkpoint"], "checkpoint_sha256": file_sha256(ck),
             "checkpoint_bytes": ck.stat().st_size}
    if ad is not None:
        ident.update({"adapter": spec["adapter"], "adapter_sha256": file_sha256(ad), "adapter_bytes": ad.stat().st_size})
        rc = ad.parent / "receipt.json"
    else:
        rc = ck.parent / "receipt.json"
    if rc.exists():
        ident["training_receipt"] = json.loads(rc.read_text()).get("receipt")
    name = "base" if ad is None else "base+lora"
    return ModelGenerator(model, res, name), ident


def _side(spec, root, split, res):
    from berkelium_neural.text3d.evaluate import evaluate
    try:
        gen, ident = _generator(spec, root, res)
    except FileNotFoundError as e:
        return {"status": "not_run", "reason": str(e), "identity": spec,
                "measurements": [m(k, status="not_run", reason="checkpoint missing", group="quality")
                                 for k, _, _ in METRICS]}
    t0 = time.time()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ev = evaluate(gen, root, split, log=print)
    agg = ev["aggregate"]
    meas = []
    for k, unit, hib in METRICS:
        v = agg[k]["mean"] if isinstance(agg[k], dict) else agg[k]
        extra = agg[k] if isinstance(agg[k], dict) else {}
        meas.append(m(k, v, unit, group="performance" if k == "seconds" else "quality", higher_is_better=hib,
                      status="measured" if v is not None else "failed",
                      reason="" if v is not None else "no successful generations to measure", **extra))
    meas.append(m("lpips", status="not_run", reason=agg["lpips"]["reason"], group="quality"))
    meas.append(m("peak_rss", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "MiB",
                  group="performance", method="process peak RSS (cumulative upper bound)",
                  delta_during_eval_mib=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1024))
    return {"status": "measured", "generator": gen.name, "identity": ident, "wall_s": round(time.time() - t0, 1),
            "split_hash": ev["split_hash"], "records": ev["records"], "measurements": meas}


def run(cfg: dict) -> dict:
    root = ROOT / cfg["dataset"]
    card = json.loads((root / "dataset_card.json").read_text())
    out = {"benchmark": "text3d", "protocol": "berkelium-text3d-eval/1", **run_metadata(cfg, "benchmarks/text3d/"
           "bench_text3d.py"), "dataset": {"name": card["dataset"], "manifest_sha256": card["manifest_sha256"],
                                           "split_hashes": card["split_hashes"]}, "comparisons": []}
    for c in cfg["comparisons"]:
        print(f"[{c['id']}] split={c['split']}")
        out["comparisons"].append({"id": c["id"], "split": c["split"],
                                   "baseline": _side(c["baseline"], root, c["split"], cfg["resolution"]),
                                   "candidate": _side(c["candidate"], root, c["split"], cfg["resolution"])})
    return out
