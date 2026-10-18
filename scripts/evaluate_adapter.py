#!/usr/bin/env python3
"""Evaluate a saved checkpoint (optionally + adapter) on a dataset split with the fixed protocol.

  python scripts/evaluate_adapter.py --checkpoint runs/textshape_base_v1/model.pt --split test
  python scripts/evaluate_adapter.py --checkpoint runs/textshape_base_v1/model.pt \
      --adapter runs/textshape_lora_capsule_v1/adapter.pt --split adapter_test
  python scripts/evaluate_adapter.py --baseline retrieval --split test

Writes results/text3d/eval_<name>_<split>.json (aggregate + every record).
"""
import argparse
import json
import sys
from pathlib import Path

from _common import ROOT, provenance


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/procedural_shapes_v1")
    ap.add_argument("--checkpoint")
    ap.add_argument("--adapter")
    ap.add_argument("--baseline", choices=["retrieval"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from berkelium_neural.text3d.dataset import load_manifest
    from berkelium_neural.text3d.evaluate import (
        ModelGenerator,
        RetrievalBaseline,
        evaluate,
    )
    from berkelium_neural.text3d.train import load_model
    root = ROOT / a.dataset
    if a.baseline:
        gen = RetrievalBaseline(load_manifest(root, {"train"} | ({"adapter_train"} if a.split.startswith("adapter")
                                                                  else set())), a.resolution)
        name = "retrieval"
    elif a.checkpoint:
        gen = ModelGenerator(load_model(ROOT / a.checkpoint, ROOT / a.adapter if a.adapter else None), a.resolution,
                             "base+lora" if a.adapter else "base")
        name = gen.name.replace("+", "_")
    else:
        ap.error("--checkpoint or --baseline required")
    res = evaluate(gen, root, a.split, a.limit)
    res["provenance"] = provenance()
    res["inputs"] = {"checkpoint": a.checkpoint, "adapter": a.adapter, "resolution": a.resolution}
    out = Path(a.out) if a.out else ROOT / "results" / "text3d" / f"eval_{name}_{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    agg = res["aggregate"]
    print(json.dumps({k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in agg.items()}, indent=2))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
