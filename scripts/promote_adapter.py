#!/usr/bin/env python3
"""E-T4: does capsule adapter v2 (replay + separation margin) fix v1's part merging without losing its geometry?

  python scripts/promote_adapter.py --candidate runs/textshape_lora_capsule_v2_replay/adapter.pt \
      --config configs/adapters/textshape_lora_capsule_v2_replay.json

Same protocol as E-T1/E-T2 (berkelium-text3d-eval/1, resolution 48): base, capsule-v1 and the candidate on
adapter_test (held-out capsule families), and v1 vs candidate on test (base prompts; retention when the adapter is
forced on). Paired 95 % bootstrap CIs over assets. The promotion rule is read from the candidate's config, where it
was written before training, and applied mechanically: the decision in the result is the rule's, not a judgement.
Writes results/text3d/E-T4_capsule_replay.json and .report.md.
"""
import argparse
import json
import sys

import numpy as np
from _common import ROOT, provenance

BASE = ROOT / "exports" / "textshape-base-v1" / "model.pt"
V1 = ROOT / "exports" / "textshape-lora-capsule-v1" / "adapter.pt"


def paired(a_rows, b_rows, key):
    """b - a over assets present in both, with a 95 % percentile bootstrap CI (10 000 resamples, seed 0)."""
    a = {r["asset_id"]: r.get(key) for r in a_rows}
    d = np.array([r[key] - a[r["asset_id"]] for r in b_rows
                  if r.get(key) is not None and a.get(r["asset_id"]) is not None], np.float64)
    if not len(d):
        return {"n": 0, "mean": None, "ci95": None}
    bs = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
    return {"n": int(len(d)), "mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                float(np.percentile(bs, 97.5))]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="data/procedural_shapes_v1")
    ap.add_argument("--out", default="results/text3d/E-T4_capsule_replay.json")
    a = ap.parse_args(argv)
    from berkelium_neural.text3d.evaluate import ModelGenerator, evaluate
    from berkelium_neural.text3d.train import file_sha256, load_model
    root = ROOT / a.dataset
    cfg = json.loads((ROOT / a.config).read_text())
    cand = ROOT / a.candidate
    gens = {"base": ModelGenerator(load_model(BASE), 48, "base"),
            "capsule-v1": ModelGenerator(load_model(BASE, V1), 48, "base+capsule-v1"),
            "candidate": ModelGenerator(load_model(BASE, cand), 48, "base+" + cfg["name"])}
    ev = {(g, s): evaluate(gens[g], root, s) for g in gens for s in ("adapter_test", "test") if not
          (g == "base" and s == "test")}
    R = lambda g, s: ev[(g, s)]["records"]                          # noqa: E731
    agg = {f"{g}|{s}": {k: (v["mean"] if isinstance(v, dict) and "mean" in v else v)
                        for k, v in e["aggregate"].items() if k != "lpips"} for (g, s), e in ev.items()}
    cmp = {"candidate_vs_v1_adapter_test": {k: paired(R("capsule-v1", "adapter_test"), R("candidate", "adapter_test"),
                                                        k) for k in ("chamfer_l1", "fscore@0.05", "view_psnr")},
           "candidate_vs_base_adapter_test": {k: paired(R("base", "adapter_test"), R("candidate", "adapter_test"), k)
                                              for k in ("chamfer_l1", "fscore@0.05", "view_psnr")},
           "candidate_vs_v1_test_forced_on": {k: paired(R("capsule-v1", "test"), R("candidate", "test"), k)
                                              for k in ("chamfer_l1", "fscore@0.05", "view_psnr")}}
    pc = agg["candidate|adapter_test"]["part_count_accuracy"]
    c_v1 = cmp["candidate_vs_v1_adapter_test"]["chamfer_l1"]["ci95"]
    c_base = cmp["candidate_vs_base_adapter_test"]["chamfer_l1"]["ci95"]
    checks = {"part_count_accuracy >= 0.53": bool(pc >= 0.53),
              "chamfer vs v1: CI upper <= +0.010": bool(c_v1 and c_v1[1] <= 0.010),
              "chamfer vs base: CI upper < 0": bool(c_base and c_base[1] < 0)}
    res = {"experiment": "E-T4", "title": "capsule adapter v2 (replay + separation margin) vs v1 and base",
           "protocol": ev[("base", "adapter_test")]["protocol"], "resolution": 48,
           "split_hashes": {s: ev[("capsule-v1", s)]["split_hash"] for s in ("adapter_test", "test")},
           "weights": {"base": file_sha256(BASE), "capsule-v1": file_sha256(V1), "candidate": file_sha256(cand)},
           "candidate_config": a.config, "promotion_rule": cfg["promotion_rule"], "aggregate": agg,
           "paired": cmp, "checks": checks, "decision": "promote" if all(checks.values()) else "keep capsule-v1",
           "environment": provenance(),
           "records": {f"{g}|{s}": e["records"] for (g, s), e in ev.items()}}
    out = ROOT / a.out
    out.write_text(json.dumps(res, indent=2, default=str))
    f = lambda x, n=3: "—" if x is None else f"{x:.{n}f}"           # noqa: E731
    L = ["# E-T4 — capsule adapter v2 (experience replay + separation margin)", "",
         f"Protocol {res['protocol']} at resolution 48; weights sha256 base `{res['weights']['base'][:12]}`, v1 "
         f"`{res['weights']['capsule-v1'][:12]}`, candidate `{res['weights']['candidate'][:12]}`. Rule written in "
         f"`{a.config}` before training.", "",
         "| model | split | n | part-count acc | CD-L1 | F@0.05 | view PSNR | colour acc |",
         "|---|---|---|---|---|---|---|---|"]
    for k, v in agg.items():
        g, s = k.split("|")
        L.append(f"| {g} | {s} | {v['n_records']} | {f(v['part_count_accuracy'], 2)} | {f(v['chamfer_l1'])} | "
                 f"{f(v['fscore@0.05'])} | {f(v['view_psnr'], 2)} | {f(v['color_accuracy'], 2)} |")
    L += ["", "| paired comparison (b − a) | metric | n | mean | 95% CI |", "|---|---|---|---|---|"]
    for name, d in cmp.items():
        for k, v in d.items():
            ci = "—" if v["ci95"] is None else f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
            L.append(f"| {name} | {k} | {v['n']} | {f(v['mean'], 4)} | {ci} |")
    L += ["", "Pre-registered checks: " + "; ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in checks.items()),
          "", f"**Decision (by the rule): {res['decision']}.**", ""]
    out.with_suffix("").with_suffix(".report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-4:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
