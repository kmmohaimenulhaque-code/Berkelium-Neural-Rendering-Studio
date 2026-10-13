#!/usr/bin/env python3
"""Build and validate a permitted dataset from a configs/datasets/*.json spec.

  python scripts/prepare_dataset.py --config configs/datasets/procedural_shapes_v1.json --out data/procedural_shapes_v1

Writes manifest.jsonl (asset id, family, split, license, source, asset sha256, caption + provenance, cameras, files),
gt/*.npz meshes, images/*.png multiview renders and dataset_card.json (split hashes, leakage check, file hashes).
Refuses to overwrite an existing dataset whose card differs (frozen splits).
"""
import argparse
import json
import sys
from pathlib import Path

from _common import load_json, provenance


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="rebuild even if a dataset already exists at --out")
    a = ap.parse_args(argv)
    from berkelium_neural.text3d.dataset import build
    cfg = load_json(a.config)
    out = Path(a.out)
    card_p = out / "dataset_card.json"
    if card_p.exists() and not a.force:
        old = json.loads(card_p.read_text())
        if old["config"] != cfg:
            print(f"refusing: {out} holds a dataset built from a different config (frozen splits). Use a new --out.")
            return 2
        print(f"dataset already built: {old['n_assets']} assets, manifest {old['manifest_sha256'][:16]}")
        return 0
    card = build(cfg, out)
    card["provenance"] = provenance()
    card_p.write_text(json.dumps(card, indent=2))
    print(json.dumps({k: card[k] for k in ("n_assets", "counts", "leakage_check", "manifest_sha256", "files_sha256",
                                            "build_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
