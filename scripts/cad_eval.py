#!/usr/bin/env python3
"""E-C1 evaluation of the design compiler: zero-shot Qwen3-8B ("base") vs Qwen3-8B + cad LoRA ("m2").

  python scripts/cad_eval.py plan     --data data/cad_v1 --out results/cad/E-C1 --n-test 480
  CUDA_VISIBLE_DEVICES=0 python scripts/cad_eval.py generate --out results/cad/E-C1 --model base --shard 0/2
  python scripts/cad_eval.py score    --data data/cad_v1 --out results/cad/E-C1 --workers 4
  python scripts/cad_eval.py report   --out results/cad/E-C1           # applies D-033, writes the report

Generation: NF4 4-bit base (same quantisation as training; a Kaggle T4 cannot hold fp16 8B), greedy, thinking off,
left-padded batches sorted by prompt length. Scoring is done by the Berkelium core only:

  parse      the completion contains one JSON object
  schema     it validates as a DesignProposal (extra fields such as measurements are rejected by the schema)
  realise    the core realises its geometry (L3 pass) with the Manifold backend
  req        every REFERENCE requirement holds on the prediction: extents for procedural parts (measured on the
             predicted geometry, the reference tolerance), equal CEM + equal requirement values for CEM rows
  iou        volume IoU of predicted vs reference solid after centring both bounding boxes (0 if not realised)"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "neural"))

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().splitlines() if x.strip()]


# ------------------------------------------------------------------------------------------------------- plan
def cmd_plan(a) -> int:
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    test = load(data / "test.jsonl.gz")
    by_fam = defaultdict(list)
    for r in test:
        by_fam[r["family"]].append(r)
    per = max(1, a.n_test // max(1, len(by_fam)))
    picked = []
    for fam in sorted(by_fam):
        rs = sorted(by_fam[fam], key=lambda r: r["id"])
        rng.shuffle(rs)
        picked += [dict(r, set="test") for r in rs[:per]]
    held = [dict(r, set="heldout") for r in load(data / "heldout.jsonl.gz")]
    rows = [{"id": r["id"], "set": r["set"], "family": r["family"], "vague": r.get("vague", False),
             "text": r["text"]} for r in picked + held]
    (out / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} items: test {len(picked)} ({len(by_fam)} families), heldout {len(held)}")
    return 0


# --------------------------------------------------------------------------------------------------- generate
def cmd_generate(a) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from berkelium.ai.prompts import cem_catalogue
    from berkelium.cem.protocol import default_registry
    from berkelium_neural.cad.prompts import PROMPT_VERSION, messages
    out = Path(a.out)
    k, n = (int(x) for x in a.shard.split("/"))
    items = [json.loads(x) for x in (out / "items.jsonl").read_text().splitlines()][k::n]
    dst = out / f"gen_{a.model}_{k}of{n}.jsonl"
    done = {json.loads(x)["id"] + json.loads(x)["set"] for x in dst.read_text().splitlines()} if dst.exists() else set()
    items = [i for i in items if i["id"] + i["set"] not in done]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    tok.pad_token = tok.pad_token or "<|endoftext|>"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, quantization_config=bnb,
                                                 dtype=torch.float16, device_map={"": 0})
    if a.model != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    cat = cem_catalogue(default_registry())
    prompts = [(i, tok.apply_chat_template(messages(i["text"], cat), tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)) for i in items]
    prompts.sort(key=lambda p: len(p[1]))
    end_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    t0 = time.time()
    with open(dst, "a") as fh:
        for s in range(0, len(prompts), a.batch):
            chunk = prompts[s:s + a.batch]
            enc = tok([p for _, p in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(0)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, eos_token_id=end_ids,
                                   pad_token_id=tok.pad_token_id)
            for (item, _), seq in zip(chunk, g[:, enc["input_ids"].shape[1]:], strict=True):
                ids = seq.tolist()
                cut = next((j for j, t in enumerate(ids) if t in end_ids), len(ids))
                fh.write(json.dumps({"id": item["id"], "set": item["set"], "model": a.model,
                                     "prompt_version": PROMPT_VERSION, "new_tokens": cut, "truncated": cut == len(ids),
                                     "completion": tok.decode(ids[:cut], skip_special_tokens=True)}) + "\n")
            fh.flush()
            print(f"[{a.model} {k}/{n}] {s + len(chunk)}/{len(prompts)} {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------ score
def _solid(record):
    """Union of every realised component's geometry (Manifold), or None."""
    from berkelium.geometry.manifold_backend import ManifoldBackend
    mb, solid = ManifoldBackend(), None
    for rc in record.realized:
        if rc.geometry is None:
            continue
        real = mb.execute(rc.geometry)
        for body in real.bodies.values():
            solid = body if solid is None else solid + body
    return solid


def _centred(m):
    bb = m.bounding_box()
    return m.translate(tuple(-(bb[i] + bb[i + 3]) / 2 for i in range(3)))


def _req_ok(ref_target: dict, pred: dict, extents: dict | None) -> bool | None:
    from berkelium.units import Quantity
    comps = ref_target["structure"]["components"]
    if comps[0]["kind"] == "cem":
        pc = [c for c in pred.get("structure", {}).get("components", []) if c.get("kind") == "cem"]
        if not pc or pc[0].get("cem") != comps[0]["cem"]:
            return False
        ref_r, got = comps[0].get("requirements", {}), pc[0].get("requirements", {})
        for key, v in ref_r.items():
            g = got.get(key)
            if isinstance(v, dict):
                try:
                    if not isinstance(g, dict) or not math.isclose(Quantity.of(g["value"], g["unit"]).si,
                                                                   Quantity.of(v["value"], v["unit"]).si,
                                                                   rel_tol=1e-4):
                        return False
                except Exception:
                    return False
            elif not (isinstance(g, int | float) and math.isclose(float(g), float(v), rel_tol=1e-4)):
                return False
        return True
    reqs = [r for r in ref_target["specification"].get("requirements", []) if r["quantity"].startswith("body.extent_")]
    if not reqs:
        return None
    if extents is None:
        return False
    for r in reqs:
        ax = r["quantity"][-1]
        tgt = Quantity.of(r["target"]["value"], r["target"]["unit"]).to("mm")
        tol = Quantity.of(r["tolerance"]["value"], r["tolerance"]["unit"]).to("mm")
        if abs(extents[ax] - tgt) > tol:
            return False
    return True


def score_one(args):
    gen, ref = args
    from berkelium.pipeline import run

    from berkelium_neural.cad.prompts import extract_json
    row = {k: gen[k] for k in ("id", "set", "model", "new_tokens", "truncated")} | {
        "family": ref["family"], "vague": ref.get("vague", False),
        "parse": False, "schema": False, "realise": False, "req": None, "iou": 0.0}
    pred = extract_json(gen["completion"])
    if pred is None:
        return row
    row["parse"] = True
    try:
        res = run(pred, realize=True, backend="manifold")
    except Exception as e:
        row["error"] = f"core exception {type(e).__name__}"
        return row
    if res.record is None:
        row["req"] = False if _req_ok(ref["target"], {}, None) is not None else None
        return row
    row["schema"] = True
    rec = res.record
    v = rec.evaluation.validation
    row["summary"] = v.summary
    l3 = [r for r in v.results if r.level == 3]
    row["realise"] = any(r.status == "pass" for r in l3) and not any(r.status in ("fail", "error") for r in l3)
    try:
        ps = _solid(rec) if row["realise"] else None
    except Exception:
        ps = None
    extents = None
    if ps is not None:
        bb = ps.bounding_box()
        extents = {ax: bb[i + 3] - bb[i] for i, ax in enumerate("xyz")}
    row["req"] = _req_ok(ref["target"], pred, extents)
    if ps is not None:
        try:
            rres = run(ref["target"], realize=True, backend="manifold")
            rs = _solid(rres.record)
            a_, b_ = _centred(ps), _centred(rs)
            inter = a_ ^ b_
            iv = 0.0 if inter.is_empty() else inter.volume()
            uv = (a_ + b_).volume()
            row["iou"] = round(float(iv / uv), 4) if uv > 0 else 0.0
        except Exception as e:
            row["iou_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return row


def cmd_score(a) -> int:
    from multiprocessing import Pool
    data, out = Path(a.data), Path(a.out)
    refs = {}
    for split in ("test", "heldout"):
        for r in load(data / f"{split}.jsonl.gz"):
            refs[(r["id"], "heldout" if split == "heldout" else "test")] = r
    gens = []
    for f in sorted(out.glob("gen_*.jsonl")):
        gens += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    jobs = [(g, refs[(g["id"], g["set"])]) for g in gens]
    with Pool(a.workers) as pool:
        rows = list(pool.imap_unordered(score_one, jobs, chunksize=4))
    rows.sort(key=lambda r: (r["model"], r["set"], r["id"]))
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"scored {len(rows)} generations")
    return 0


# ----------------------------------------------------------------------------------------------------- report
def _agg(rows):
    n = len(rows)
    if not n:
        return {}
    rq = [r["req"] for r in rows if r["req"] is not None]
    return {"n": n, "parse": round(sum(r["parse"] for r in rows) / n, 4),
            "schema": round(sum(r["schema"] for r in rows) / n, 4),
            "realise": round(sum(r["realise"] for r in rows) / n, 4),
            "req": round(sum(rq) / len(rq), 4) if rq else None, "req_n": len(rq),
            "iou": round(sum(r["iou"] for r in rows) / n, 4),
            "truncated": round(sum(r["truncated"] for r in rows) / n, 4)}


def _boot_ci(diffs, n_boot=4000, seed=0):
    rng = random.Random(seed)
    m = len(diffs)
    if m == 0:
        return None
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_boot))
    return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4)]


def cmd_report(a) -> int:
    out = Path(a.out)
    rows = [json.loads(x) for x in (out / "scores.jsonl").read_text().splitlines() if x.strip()]
    res = {"models": {}, "families": {}}
    for m in sorted({r["model"] for r in rows}):
        res["models"][m] = {}
        for s in ("test", "heldout"):
            rs = [r for r in rows if r["model"] == m and r["set"] == s]
            res["models"][m][s] = _agg(rs)
            res["models"][m][s + "_gear"] = _agg([r for r in rs if r["family"] == "gear_pair"])
        for fam in sorted({r["family"] for r in rows}):
            res["families"].setdefault(fam, {})[m] = _agg([r for r in rows if r["model"] == m and r["family"] == fam])
    paired = {}
    if {"base", "m2"} <= set(res["models"]):
        for s in ("test", "heldout"):
            b = {r["id"]: r for r in rows if r["model"] == "base" and r["set"] == s}
            m2 = {r["id"]: r for r in rows if r["model"] == "m2" and r["set"] == s}
            ids = sorted(set(b) & set(m2))
            d = [m2[i]["iou"] - b[i]["iou"] for i in ids]
            paired[s] = {"n": len(ids), "iou_diff_mean": round(sum(d) / max(1, len(d)), 4), "ci95": _boot_ci(d)}
    res["paired_iou_m2_minus_base"] = paired
    # D-033 (pre-registered 2026-10-17, NEURAL_RENDERING_DECISIONS.md)
    m2, base = res["models"].get("m2", {}), res["models"].get("base", {})
    checks = {}
    if m2:
        t, h = m2["test"], m2["heldout"]
        checks = {
            "test schema >= 0.95": t["schema"] >= 0.95,
            "test realise >= 0.90": t["realise"] >= 0.90,
            "test req >= 0.85": (t["req"] or 0) >= 0.85,
            "heldout realise >= 0.60": h["realise"] >= 0.60,
            "heldout IoU beats base (paired CI > 0)": bool(paired.get("heldout", {}).get("ci95"))
            and paired["heldout"]["ci95"][0] > 0,
            "gear req not below base": (m2["test_gear"].get("req") or 0) >= (base.get("test_gear", {}).get("req") or 0),
        }
    res["d033"] = {"checks": checks, "promote": bool(checks) and all(checks.values())}
    (out / "E-C1.json").write_text(json.dumps(res, indent=2))
    lines = ["# E-C1: text -> DesignProposal compiler (Qwen3-8B, NF4 on Kaggle T4)", "",
             "Scored by the Berkelium core (Manifold backend). Rule: D-033 (pre-registered).", "",
             "| model | set | n | parse | schema | realise | req | IoU | truncated |", "|---|---|---|---|---|---|---|---|---|"]
    for m, sets in res["models"].items():
        for s in ("test", "heldout", "test_gear"):
            x = sets.get(s) or {}
            if x:
                lines.append(f"| {m} | {s} | {x['n']} | {x['parse']} | {x['schema']} | {x['realise']} | {x['req']} | "
                             f"{x['iou']} | {x['truncated']} |")
    lines += ["", "Paired IoU (m2 - base): " + json.dumps(paired), "", "## D-033", ""]
    lines += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()]
    lines += ["", f"**Decision: {'PROMOTE m2' if res['d033']['promote'] else 'do not promote'}**", "",
              "## Per family (realise / req / IoU)", "", "| family | " + " | ".join(res["models"]) + " |",
              "|---|" + "---|" * len(res["models"])]
    for fam, ms in res["families"].items():
        lines.append(f"| {fam} | " + " | ".join(f"{ms[m]['realise']} / {ms[m]['req']} / {ms[m]['iou']}" if ms.get(m)
                                                 else "-" for m in res["models"]) + " |")
    (out / "E-C1.report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--data", default="data/cad_v1")
    p.add_argument("--out", default="results/cad/E-C1")
    p.add_argument("--n-test", type=int, default=480)
    g = sub.add_parser("generate")
    g.add_argument("--out", default="results/cad/E-C1")
    g.add_argument("--model", required=True)
    g.add_argument("--adapter", default="exports/qwen3-8b-cad-v1/adapter")
    g.add_argument("--shard", default="0/1")
    g.add_argument("--batch", type=int, default=12)
    g.add_argument("--max-new", type=int, default=2048)
    s = sub.add_parser("score")
    s.add_argument("--data", default="data/cad_v1")
    s.add_argument("--out", default="results/cad/E-C1")
    s.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    r = sub.add_parser("report")
    r.add_argument("--out", default="results/cad/E-C1")
    a = ap.parse_args()
    return {"plan": cmd_plan, "generate": cmd_generate, "score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
