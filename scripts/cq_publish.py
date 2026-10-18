#!/usr/bin/env python3
"""Publish the E-C2 adapter and the merged bf16 model to Hugging Face, with a model card built from E-C2.json, and
put the results into the repo tree.

  python scripts/cq_publish.py --train ~/berkelium_run/train --results results/cadquery

Repos: ``Mhaquehaque/berkelium-qwen3-<32b|14b>-cadquery-v1`` (LoRA adapter + run manifest + E-C2 results) and
``...-cadquery-v1-merged`` (base + adapter merged, bf16 safetensors). Private unless ``--public``. The token is read
from ``HF_TOKEN`` only and is never printed. The card states the D-034 verdict exactly as cq_eval.py wrote it.

Results: E-C2.json / E-C2.report.md are already under ``results/cadquery/`` (cq_eval.py writes there); this script
adds ``publish_receipt.json`` (repo ids, revisions, file SHA-256) and commits them on the current branch. It
pushes only when ``BERKELIUM_GIT_PUSH=1``; otherwise it writes ``~/berkelium_run/results.bundle`` for the owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "cadquery")]

import cqformat  # noqa: E402
import pins  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


NOT_EVALUATED = {"d034": {"verdict": "NOT EVALUATED", "criteria": [],
                          "action": "ship as opt-in; E-C2 eval was skipped (owner decision, deadline), so D-034 was not applied"},
                 "host": {"evidence": "no evaluation run"}, "generation_backend": "-", "finished_utc": "-",
                 "tuned": {"metrics": {}}, "base": {"metrics": {}}}


def card(base: str, res: dict, merged: bool, repo: str) -> str:
    rev = pins.MODELS[base]["revision"]
    d = res["d034"]
    rows = []
    for g, t in res["tuned"]["metrics"].items():
        b = res["base"]["metrics"][g]
        if t["first"].get("n"):
            rows.append(f"| {g} | {t['first']['n']} | {b['first']['execute_valid']:.3f} | {t['first']['execute_valid']:.3f} "
                        f"| {b['first']['iou_mean']:.3f} | {t['first']['iou_mean']:.3f} | {t['with_repair']['execute_valid']:.3f} |")
    crit = "\n".join(f"| {c['name']} | {c['value']} | {c['threshold']} | {'PASS' if c['pass'] else 'FAIL'} |"
                     for c in d["criteria"])
    lic = "apache-2.0"
    return f"""---
license: other
license_name: see-card
base_model: {base}
tags: [cadquery, cad, text-to-cad, berkelium, lora]
---
# {repo.split('/')[-1]}

Text → CadQuery compiler for Berkelium Neural Rendering Studio: {'the LoRA adapter merged into' if merged else 'a LoRA adapter (r=64, all linear layers) for'}
`{base}` @ `{rev}`. It writes one JSON header line and one CadQuery script in millimetres ending with
`result = ...`. Thinking off. System prompt (as trained):

> {cqformat.SYSTEM}

**The model only proposes.** Execution happens in a sandbox (AST whitelist + isolated process), and geometry,
measurements and validity come from OCCT through the deterministic Berkelium core
({pins.CORE['repo']} @ `{pins.CORE['commit'][:7]}`).

## E-C2 evaluation (held-out data; {res['host']['evidence']}; {res['generation_backend']}; {res['finished_utc']})

| group | n | base exec+valid | tuned exec+valid | base IoU | tuned IoU | tuned exec+valid (≤2 repairs) |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

### D-034 (pre-registered before training)

| criterion | value | threshold | pass |
|---|---|---|---|
{crit}

**Verdict: {d['verdict']}** — {d['action']}.

## Training data (berkelium-cq-v1)
Sandbox-verified rows from: Text-to-CadQuery (arXiv 2505.06507; `{pins.SOURCES['text2cadquery']['hf_repo']}` @
`{pins.SOURCES['text2cadquery']['revision'][:7]}`, licence {pins.SOURCES['text2cadquery']['license']}), Berkelium
parametric families transpiled from Geometry IR, repair rows, and CAD-Recode v1.5 (arXiv 2412.14042,
`{pins.SOURCES['cad_recode']['hf_repo']}`, **{pins.SOURCES['cad_recode']['license']}**) with request texts written
by the base model. Because CAD-Recode is CC-BY-NC-4.0, treat this model as **non-commercial** unless it is
retrained without that source. Base model licence: {lic}.

Held-out families never trained on: {', '.join(pins.HELDOUT_FAMILIES)}.
"""


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=check)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default="~/berkelium_run/train")
    ap.add_argument("--results", default=str(ROOT / "results" / "cadquery"))
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--no-merged", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the cards locally, upload nothing")
    a = ap.parse_args()
    train = Path(a.train).expanduser()
    final = json.loads((train / "final.json").read_text())
    base, adapter = final["model"], Path(final["adapter"])
    size = base.split("-")[-1].lower()
    repo = pins.HF_ADAPTER.format(size=size)
    rd = Path(a.results)
    rd.mkdir(parents=True, exist_ok=True)
    res = json.loads((rd / "E-C2.json").read_text()) if (rd / "E-C2.json").exists() else NOT_EVALUATED
    run_dir = adapter.parent
    (adapter / "README.md").write_text(card(base, res, False, repo))
    for f in ("run_manifest.json", "status.json"):
        if (run_dir / f).exists():
            (adapter / f).write_text((run_dir / f).read_text())
    if (rd / "E-C2.json").exists():
        (adapter / "E-C2.json").write_text((rd / "E-C2.json").read_text())
    receipt = {"published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base": base,
               "base_revision": pins.MODELS[base]["revision"], "d034": res["d034"]["verdict"],
               "adapter_repo": repo, "adapter_files": {p.name: sha256_file(p) for p in adapter.iterdir() if p.is_file()}}
    if a.dry_run:
        print(json.dumps(receipt, indent=1))
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo, private=not a.public, exist_ok=True)
    c = api.upload_folder(repo_id=repo, folder_path=str(adapter), commit_message=f"E-C2 adapter ({res['d034']['verdict']})")
    receipt["adapter_commit"] = getattr(c, "oid", None) or str(c)
    if not a.no_merged:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mdir = train / f"merged-{size}"
        if not (mdir / "config.json").exists():
            m = AutoModelForCausalLM.from_pretrained(base, revision=pins.MODELS[base]["revision"], dtype=torch.bfloat16,
                                                     device_map="cpu")
            m = PeftModel.from_pretrained(m, str(adapter)).merge_and_unload()
            m.save_pretrained(str(mdir), safe_serialization=True, max_shard_size="5GB")
            AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(str(mdir))
        (mdir / "README.md").write_text(card(base, res, True, repo + "-merged"))
        api.create_repo(repo + "-merged", private=not a.public, exist_ok=True)
        c2 = api.upload_folder(repo_id=repo + "-merged", folder_path=str(mdir),
                               commit_message=f"E-C2 merged bf16 ({res['d034']['verdict']})")
        receipt.update(merged_repo=repo + "-merged", merged_commit=getattr(c2, "oid", None) or str(c2),
                       merged_files={p.name: sha256_file(p) for p in mdir.iterdir() if p.is_file() and p.suffix == ".json"})
    (rd / "publish_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))
    # ---- results into the repo tree
    git("add", str(rd))
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        git("commit", "-q", "-m", f"E-C2 results ({res['d034']['verdict']}) and publish receipt")
    if os.environ.get("BERKELIUM_GIT_PUSH") == "1":
        p = git("push", "origin", "HEAD", check=False)
        print("git push:", "ok" if p.returncode == 0 else p.stderr.strip()[-300:])
    else:
        b = Path("~/berkelium_run/results.bundle").expanduser()
        git("bundle", "create", str(b), "origin/universal-cadquery..HEAD", check=False)
        print(f"results committed locally; bundle for the owner: {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
