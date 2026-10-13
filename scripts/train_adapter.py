#!/usr/bin/env python3
"""Execute a configured training job (base model or LoRA adapter) from configs/adapters/*.json.

  python scripts/train_adapter.py --config configs/adapters/textshape_base_v1.json
  python scripts/train_adapter.py --config configs/adapters/textshape_lora_capsule_v1.json

Resumable: rerunning the same command continues from runs/<name>/checkpoint.pt (refuses if the config changed).
--max-steps N stops this invocation after N more steps (checkpointed) for bounded sessions.
Device: CPU here. The model is small; GPU execution is not required for this pathway (record says gpu_hours 0).
"""
import argparse
import json
import sys

from _common import load_json, provenance


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args(argv)
    from berkelium_neural.text3d.train import train
    cfg = load_json(a.config)
    if a.max_steps:
        cfg["max_steps_this_run"] = a.max_steps
    res = train(cfg)
    res["provenance"] = provenance()
    print(json.dumps({k: res[k] for k in res if k != "provenance"}, indent=2))
    return 0 if res.get("step", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
