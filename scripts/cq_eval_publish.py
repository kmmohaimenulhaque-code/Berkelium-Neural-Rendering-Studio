#!/usr/bin/env python3
"""After cq_eval.py: write a strengths/limitations report with real examples from the per-item records, and add
it (plus E-C2.json and the D-034 verdict) to the three public HF repos' model cards.

  python scripts/cq_eval_publish.py --results results/cadquery --items ~/berkelium_run/eval/E-C2_items.jsonl.gz

Every number and example comes from E-C2.json / E-C2_items.jsonl.gz. A group needs ≥ 5 items for a rate claim.
The GGUF repo's card says the numbers are for the bf16 model; the Q4_K_M quantization itself was not evaluated."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- E-C2:BEGIN -->", "<!-- E-C2:END -->"


def load_items(p: Path) -> list[dict]:
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def short(code: str | None, n: int = 18) -> str:
    lines = (code or "").strip().splitlines()
    return "\n".join(lines[:n]) + ("\n# …" if len(lines) > n else "")


def analyse(items: list[dict], data_dir: Path) -> str:
    texts = {}
    for split in ("test", "heldout"):
        p = data_dir / f"{split}.jsonl.gz"
        if p.exists():
            with gzip.open(p, "rt") as f:
                for x in f:
                    r = json.loads(x)
                    texts[r["id"]] = r["text"]
    tuned = [r for r in items if r["model"] == "tuned"]
    base = {r["id"]: r for r in items if r["model"] == "base"}
    groups = defaultdict(list)
    for r in tuned:
        groups[f"{r['source']}:{r['family']}" if r.get("family") else r["source"]].append(r)
    rows = []
    for g, rs in groups.items():
        if len(rs) < 5:
            continue
        ok = sum(r["first"]["ok"] for r in rs) / len(rs)
        iou = sum(r["first"].get("iou", 0) for r in rs) / len(rs)
        bok = sum(base[r["id"]]["first"]["ok"] for r in rs if r["id"] in base) / len(rs)
        rows.append((g, len(rs), ok, iou, bok))
    rows.sort(key=lambda x: (-x[2], -x[3]))
    L = ["## What it does well and where it fails (from the E-C2 items)", "",
         "Rates are first-attempt execute+valid (sandbox + core L3) and mean IoU vs the reference; groups with ≥ 5 items.", "",
         "| group | n | tuned exec+valid | tuned IoU | base exec+valid |", "|---|---|---|---|---|"]
    L += [f"| {g} | {n} | {ok:.2f} | {iou:.2f} | {bok:.2f} |" for g, n, ok, iou, bok in rows]
    good = sorted([r for r in tuned if r["first"]["ok"]], key=lambda r: -r["first"].get("iou", 0))[:3]
    stages = defaultdict(int)
    for r in tuned:
        if not r["first"]["ok"]:
            stages[r["first"]["stage"]] += 1
    bad = [r for r in tuned if not r["first"]["ok"]]
    bad_by_stage = {}
    for r in bad:
        bad_by_stage.setdefault(r["first"]["stage"], r)
    L += ["", "### Strong examples (highest IoU)"]
    for r in good:
        L += ["", f"**Prompt** ({r['source']}{', ' + r['family'] if r.get('family') else ''}; IoU {r['first'].get('iou', 0):.3f}):",
              f"> {texts.get(r['id'], '?')[:400]}", "", "```python", short(r["first"].get("code")), "```"]
    L += ["", "### Limitations", "", "First-attempt failures by stage: " +
          ", ".join(f"{k} {v}" for k, v in sorted(stages.items(), key=lambda x: -x[1])) + f" (of {len(tuned)})."]
    weak = [x for x in rows if x[2] < 0.6]
    if weak:
        L.append("Weakest groups: " + ", ".join(f"{g} ({ok:.2f})" for g, _, ok, _, _ in weak[-5:]) + ".")
    for st, r in list(bad_by_stage.items())[:3]:
        L += ["", f"**Failure: `{st}`** ({r['source']}{', ' + r['family'] if r.get('family') else ''}):",
              f"> {texts.get(r['id'], '?')[:300]}", "", f"Error: `{(r['first'].get('error') or '')[:300]}`"]
        if r["repaired"]["ok"]:
            L.append(f"Fixed by the repair loop in round {r['repaired'].get('round', '?')}.")
    L += ["", "General limits: millimetre CadQuery only (no assemblies with joints, no threads unless modelled); "
          "absolute placement may differ from the reference (IoU aligns bbox centres); the output is code and must "
          "run in a sandbox; numbers above are for the bf16 model."]
    return "\n".join(L)


def section(res: dict, analysis: str, gguf: bool) -> str:
    d = res["d034"]
    rows = []
    for g, t in res["tuned"]["metrics"].items():
        b = res["base"]["metrics"][g]
        if t["first"].get("n"):
            rows.append(f"| {g} | {t['first']['n']} | {b['first']['execute_valid']:.3f} | {t['first']['execute_valid']:.3f} | "
                        f"{b['first']['iou_mean']:.3f} | {t['first']['iou_mean']:.3f} | {t['with_repair']['execute_valid']:.3f} |")
    crit = "\n".join(f"| {c['name']} | {c['value']} | {c['threshold']} | {'PASS' if c['pass'] else 'FAIL'} |" for c in d["criteria"])
    note = "\n\n> These numbers are for the bf16 model. The Q4_K_M GGUF was not evaluated separately.\n" if gguf else "\n"
    if not res["base"]["metrics"]["all_heldout"]["first"].get("format_ok", 1):
        note += ("\n> The zero-shot base never produced a parseable script in the required format (format_ok 0.0), so its 0.000 "
                 "measures format compliance, not CAD ability.\n")
    return f"""{BEGIN}
## E-C2 evaluation ({res['finished_utc']}, {res['host']['device']}, {res['generation_backend']}){note}
Zero-shot `{res['base']['model']}` vs fine-tuned, same prompts, greedy, thinking off; {res['data']['n_items']} held-out prompts
(up to {res['data']['per_source_cap']} per source and split; held-out families never trained on).

| group | n | base exec+valid | tuned exec+valid | base IoU | tuned IoU | tuned exec+valid (≤2 repairs) |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

### D-034 (pre-registered before training)
| criterion | value | threshold | pass |
|---|---|---|---|
{crit}

**Verdict: {d['verdict']}**: {d['action']}.

{analysis}
{END}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "cadquery"))
    ap.add_argument("--items", default="~/berkelium_run/eval/E-C2_items.jsonl.gz")
    ap.add_argument("--data", default="~/berkelium_run/data/cq_v1")
    a = ap.parse_args()
    rd = Path(a.results)
    res = json.loads((rd / "E-C2.json").read_text())
    analysis = analyse(load_items(Path(a.items).expanduser()), Path(a.data).expanduser())
    (rd / "E-C2.examples.md").write_text(analysis + "\n")
    rec = json.loads((rd / "publish_receipt.json").read_text())
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ.get("HF_TOKEN"))   # None -> cached `hf auth login`
    for repo, gguf in ((rec["adapter_repo"], False), (rec.get("merged_repo"), False), (rec.get("gguf_repo"), True)):
        if not repo:
            continue
        card = Path(hf_hub_download(repo, "README.md", token=os.environ.get("HF_TOKEN"))).read_text()
        card = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END) + "\n?", "", card, flags=re.S)
        card = card.replace("NOT EVALUATED", f"see E-C2 below ({res['d034']['verdict']})")
        card = card.rstrip() + "\n\n" + section(res, analysis, gguf)
        tmp = rd / f".card_{repo.split('/')[-1]}.md"
        tmp.write_text(card)
        api.upload_file(path_or_fileobj=str(tmp), path_in_repo="README.md", repo_id=repo, commit_message="E-C2 evaluation")
        for f in ("E-C2.json", "E-C2.report.md", "E-C2.examples.md"):
            api.upload_file(path_or_fileobj=str(rd / f), path_in_repo=f"eval/{f}", repo_id=repo, commit_message="E-C2 evaluation")
        tmp.unlink()
        print("updated", f"https://huggingface.co/{repo}")
    print(f"D-034: {res['d034']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
