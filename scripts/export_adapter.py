#!/usr/bin/env python3
"""Export trained weights with configuration, model identity and provenance into a self-describing bundle.

  python scripts/export_adapter.py --run runs/textshape_lora_capsule_v1 --out exports/textshape-lora-capsule-v1
  python scripts/export_adapter.py --run runs/textshape_base_v1 --out exports/textshape-base-v1

Bundle: weights file (model.pt or adapter.pt), train_config.json, receipt.json, MODEL_CARD.json (identity, base binding,
dataset manifest/split hashes, license, sha256 of every file). Refuses incomplete runs.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from _common import ROOT, provenance


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from berkelium_neural.text3d.train import file_sha256
    run = ROOT / a.run
    rc = json.loads((run / "receipt.json").read_text())
    if not rc.get("complete"):
        print(f"refused: {run} is not a completed run (step {rc.get('step')})")
        return 2
    weights = Path(rc["artifact"])
    if file_sha256(weights) != rc["artifact_sha256"]:
        print("refused: weights do not match the receipt hash")
        return 3
    import torch
    st = torch.load(weights, map_location="cpu", weights_only=False)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(weights, out / weights.name)
    (out / "train_config.json").write_text(json.dumps(st["config"], indent=2))   # not config.json: gitignored
    (out / "receipt.json").write_text(json.dumps(rc, indent=2))
    card = {"name": st["config"].get("name"), "kind": st["config"]["kind"], "train_version": st.get("train_version"),
            "config_hash": st["config_hash"], "weights": weights.name, "license": "MIT (Berkelium-owned weights; "
            "trained only on Berkelium-owned procedural data)", "identity": st["config"].get("model_identity"),
            "dataset": {"manifest_sha256": rc["receipt"]["dataset_manifest_sha256"],
                        "split_hashes": rc["receipt"]["dataset_split_hashes"]},
            "export_provenance": provenance()}
    if st["config"]["kind"] == "lora":
        card["base_checkpoint_sha256"] = st["base_checkpoint_sha256"]
        card["adapter_config"] = st["adapter_config"]
    card["files"] = {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "MODEL_CARD.json"}
    (out / "MODEL_CARD.json").write_text(json.dumps(card, indent=2))
    print(json.dumps({"exported": str(out), "files": card["files"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
