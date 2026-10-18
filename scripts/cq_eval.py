#!/usr/bin/env python3
"""E-C2: zero-shot base vs fine-tuned text → CadQuery compiler on held-out data, scored by the sandbox and the
Berkelium core, and the D-034 verdict.

  python scripts/cq_eval.py --data ~/berkelium_run/data/cq_v1 --adapter ~/berkelium_run/train/qwen3-14b/adapter \
      --base Qwen/Qwen3-14B --results results/cadquery

Metric definitions (fixed here, in the commit that precedes training):

* **execute+valid**: the completion parses, the script runs in cadquery/sandbox.py and returns a solid, and the
  core's L3 check (OCCT valid + closed + volume > 0, cadquery/core_check.py) passes. First attempt only.
* **IoU**: exact OCCT volumetric IoU against the reference script's solid after aligning bounding-box centres
  (raw-frame IoU is also reported). A failed attempt scores 0. The paired comparison is fine-tuned − base on the
  same prompts (test ∪ held-out families), 10 000 bootstrap resamples, seed 0.
* **held-out-family execute**: sandbox success (runs + valid solid) on ``pipe_elbow``, ``chair``, ``knob``,
  ``hinge_leaf``, first attempt.
* **sandbox rejection**: share of cadquery/unsafe_corpus.py scripts the sandbox does not run to success.
* **with repair**: the same scores after ≤ 2 repair rounds (error fed back); reported, not part of D-034.

Sampling: up to ``--per-source`` rows per (source, split), chosen by a hash of the row id. Generation: vLLM
(ROCm) with the LoRA loaded via ``LoRARequest`` (the base is the same engine without it), greedy, thinking off;
falls back to HF transformers if vLLM cannot start (reason recorded). Writes ``E-C2.json`` and ``E-C2.report.md``."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "cadquery")]

import cqformat  # noqa: E402
import cqpipeline  # noqa: E402
import pins  # noqa: E402
import sandbox  # noqa: E402
from unsafe_corpus import UNSAFE  # noqa: E402

D034 = {"execute_valid_min": 0.85, "heldout_execute_min": 0.60, "sandbox_reject": 1.0}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def pick(rows: list[dict], per: int) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        if r.get("kind") == "gen" and r.get("text"):
            by[(r["source"], r["split"])].append(r)
    out = []
    for k, rs in sorted(by.items()):
        rs.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        out += rs[:per]
    return out


def _iou_job(args):
    import core_check
    cand, ref = args
    try:
        return core_check.iou(cand, ref, "bbox_center"), core_check.iou(cand, ref, "none")
    except Exception:  # noqa: BLE001
        return None


def guarded_iou(pairs: list[tuple[bytes, bytes]], workers: int, timeout: float = 120) -> list[tuple | None]:
    """IoU in worker processes; a pair that hangs or crashes OCCT scores None (counted as 0)."""
    out: list = [None] * len(pairs)
    todo = list(range(len(pairs)))
    while todo:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(workers, maxtasksperchild=50)
        futs = {i: pool.apply_async(_iou_job, (pairs[i],)) for i in todo}
        todo = []
        broken = False
        for i, f in futs.items():
            if broken:
                todo.append(i)
                continue
            try:
                out[i] = f.get(timeout=timeout)
            except mp.TimeoutError:
                out[i] = None
                broken = True               # restart the pool; remaining items are retried
            except Exception:  # noqa: BLE001
                out[i] = None
        pool.terminate()
        pool.join()
    return out


def bootstrap_ci(diff: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    import numpy as np
    d = np.asarray(diff, dtype=float)
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n, d.size))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_model(backend, items: list[dict], adapter: bool, runner, rounds: int) -> list[dict]:
    """Batched first attempt + repair rounds. Returns one record per item (artifacts stripped except STEP)."""
    comps = backend.chat([cqformat.messages(it["text"]) for it in items], adapter=adapter)
    recs = [cqpipeline.attempt(c, runner) for c in comps]
    first = [dict(r) for r in recs]
    final = list(recs)
    for rnd in range(1, rounds + 1):
        idx = [i for i, r in enumerate(final) if not r["ok"]]
        if not idx:
            break
        msgs = [cqformat.repair_messages(items[i]["text"], final[i]["completion"],
                                         cqpipeline.error_for_repair(final[i])) for i in idx]
        for i, c in zip(idx, backend.chat(msgs, adapter=adapter)):
            final[i] = cqpipeline.attempt(c, runner)
            final[i]["round"] = rnd
    out = []
    for it, f, z in zip(items, first, final):
        out.append({"id": it["id"], "source": it["source"], "split": it["split"], "family": it.get("family"),
                    "first": {k: v for k, v in f.items() if not k.startswith("_")}, "_first_step": f.get("_step"),
                    "repaired": {k: v for k, v in z.items() if not k.startswith("_")}, "_rep_step": z.get("_step")})
    return out


def score(recs: list[dict], ref_steps: dict[str, bytes], workers: int) -> None:
    for key in ("first", "repaired"):
        pairs, idx = [], []
        for i, r in enumerate(recs):
            st, ref = r[f"_{'first' if key == 'first' else 'rep'}_step"], ref_steps.get(r["id"])
            if r[key]["ok"] and st and ref:
                pairs.append((st, ref))
                idx.append(i)
        vals = guarded_iou(pairs, workers)
        for r in recs:
            r[key]["iou"], r[key]["iou_raw"] = 0.0, 0.0
        for i, v in zip(idx, vals):
            if v is not None:
                recs[i][key]["iou"], recs[i][key]["iou_raw"] = v
            else:
                recs[i][key]["iou_failed"] = True


def summarize(recs: list[dict]) -> dict:
    def agg(rs, key):
        n = len(rs)
        if not n:
            return {"n": 0}
        return {"n": n, "execute_valid": sum(r[key]["ok"] for r in rs) / n,
                "execute": sum(r[key]["executed"] for r in rs) / n,
                "iou_mean": sum(r[key].get("iou", 0.0) for r in rs) / n,
                "iou_raw_mean": sum(r[key].get("iou_raw", 0.0) for r in rs) / n,
                "format_ok": sum(not r[key]["format_problems"] for r in rs) / n}
    groups = {"all_heldout": recs, "test": [r for r in recs if r["split"] == "test"],
              "heldout_families": [r for r in recs if r["split"] == "heldout"]}
    for s in sorted({r["source"] for r in recs}):
        groups[f"test:{s}"] = [r for r in recs if r["split"] == "test" and r["source"] == s]
    return {g: {"first": agg(rs, "first"), "with_repair": agg(rs, "repaired")} for g, rs in groups.items()}


def report_md(res: dict) -> str:
    L = [f"# E-C2 — text → CadQuery compiler ({res['tuned']['model']})", "",
         f"Run {res['finished_utc']} on `{res['host']['device']}` (backend {res['generation_backend']}). "
         f"Decision rule D-034 (pre-registered in `NEURAL_RENDERING_DECISIONS.md`). Every number below is in "
         f"`E-C2.json`; per-item records are at `{res['items_file']}` (sha256 `{res['items_sha256'][:16]}…`).", "",
         "| group | n | base exec+valid | tuned exec+valid | base IoU | tuned IoU | tuned exec+valid (≤2 repairs) |",
         "|---|---|---|---|---|---|---|"]
    for g in res["tuned"]["metrics"]:
        b, t = res["base"]["metrics"][g], res["tuned"]["metrics"][g]
        if not t["first"].get("n"):
            continue
        L.append(f"| {g} | {t['first']['n']} | {b['first']['execute_valid']:.3f} | {t['first']['execute_valid']:.3f} | "
                 f"{b['first']['iou_mean']:.3f} | {t['first']['iou_mean']:.3f} | {t['with_repair']['execute_valid']:.3f} |")
    d = res["d034"]
    L += ["", "## D-034", "", "| criterion | value | threshold | pass |", "|---|---|---|---|"]
    for c in d["criteria"]:
        L.append(f"| {c['name']} | {c['value']} | {c['threshold']} | {'PASS' if c['pass'] else 'FAIL'} |")
    L += ["", f"**Verdict: {d['verdict']}** — {d['action']}", "",
          "Evidence class: " + res["host"]["evidence"] + ". Sandbox network isolation: " + res["sandbox"]["net_isolation"] + "."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="~/berkelium_run/data/cq_v1")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", required=True, choices=sorted(pins.MODELS))
    ap.add_argument("--results", default=str(ROOT / "results" / "cadquery"))
    ap.add_argument("--items-dir", default="~/berkelium_run/eval")
    ap.add_argument("--per-source", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=cqpipeline.REPAIR_ROUNDS)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    data = Path(a.data).expanduser()
    workers = a.workers or os.cpu_count() or 1
    items = pick(load(data / "test.jsonl.gz") + load(data / "heldout.jsonl.gz"), a.per_source)
    print(f"{len(items)} evaluation prompts", flush=True)
    from llm import make_backend
    rev = pins.MODELS[a.base]["revision"]
    backend, why = make_backend(a.base, rev, str(Path(a.adapter).expanduser()), a.backend, a.tp)
    with sandbox.BatchRunner(jobs=workers, keep_artifacts=True) as runner:
        unsafe = dict(zip(UNSAFE, runner.map(list(UNSAFE.values()))))
        rejected = sum(not r.ok for r in unsafe.values()) / len(unsafe)
        refs = {}
        for it, r in zip(items, runner.map([it["code"] for it in items])):
            if r.ok and r.step:
                refs[it["id"]] = r.step
        out = {}
        for name, use_adapter in (("base", False), ("tuned", True)):
            t1 = time.time()
            recs = run_model(backend, items, use_adapter, runner, a.rounds)
            score(recs, refs, workers)
            out[name] = {"recs": recs, "metrics": summarize(recs), "seconds": round(time.time() - t1, 1)}
            print(name, json.dumps(out[name]["metrics"]["all_heldout"]), flush=True)
    import core_check  # noqa: F401  (imported for the versions below)
    base_iou = {r["id"]: r["first"]["iou"] for r in out["base"]["recs"]}
    diff = [r["first"]["iou"] - base_iou[r["id"]] for r in out["tuned"]["recs"]]
    mean_d, lo, hi = bootstrap_ci(diff)
    tm = out["tuned"]["metrics"]
    crit = [
        {"name": "execute+valid (held-out, first attempt)", "value": round(tm["all_heldout"]["first"]["execute_valid"], 4),
         "threshold": f">= {D034['execute_valid_min']}", "pass": tm["all_heldout"]["first"]["execute_valid"] >= D034["execute_valid_min"]},
        {"name": "IoU tuned − base, paired 95 % CI", "value": f"{mean_d:.4f} [{lo:.4f}, {hi:.4f}]",
         "threshold": "CI lower > 0", "pass": lo > 0},
        {"name": "held-out-family execute", "value": round(tm["heldout_families"]["first"].get("execute", 0.0), 4),
         "threshold": f">= {D034['heldout_execute_min']}", "pass": tm["heldout_families"]["first"].get("execute", 0.0) >= D034["heldout_execute_min"]},
        {"name": "sandbox rejects injected unsafe scripts", "value": f"{rejected:.4f} ({len(unsafe)} scripts)",
         "threshold": "1.0", "pass": rejected == 1.0},
    ]
    passed = all(c["pass"] for c in crit)
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        hip = getattr(torch.version, "hip", None)
    except ImportError:
        dev, hip = "unknown", None
    items_dir = Path(a.items_dir).expanduser()
    items_dir.mkdir(parents=True, exist_ok=True)
    items_file = items_dir / "E-C2_items.jsonl.gz"
    with gzip.open(items_file, "wt") as f:
        for name in ("base", "tuned"):
            for r in out[name]["recs"]:
                f.write(json.dumps({"model": name, **{k: v for k, v in r.items() if not k.startswith("_")}},
                                   default=str) + "\n")
    adapter = Path(a.adapter).expanduser()
    res = {"experiment": "E-C2", "decision": "D-034", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seconds": round(time.time() - t0, 1), "generation_backend": backend.name, "vllm_fallback_reason": why,
           "host": {"device": dev, "hip": hip, "platform": platform.platform(),
                    "evidence": "ROCm/AMD" if hip else "CUDA or CPU (not ROCm evidence)"},
           "data": {"dir": str(data), "manifest_sha256": sha256_file(data / "manifest.json") if (data / "manifest.json").exists() else None,
                    "n_items": len(items), "n_refs": len(refs), "per_source_cap": a.per_source},
           "base": {"model": a.base, "revision": rev, "metrics": out["base"]["metrics"], "seconds": out["base"]["seconds"]},
           "tuned": {"model": f"{a.base} + LoRA", "adapter": str(adapter),
                     "adapter_sha256": {p.name: sha256_file(p) for p in adapter.iterdir() if p.is_file()} if adapter.is_dir() else None,
                     "metrics": tm, "seconds": out["tuned"]["seconds"]},
           "iou_paired": {"mean_diff": mean_d, "ci95": [lo, hi], "n": len(diff), "bootstrap": 10000, "seed": 0},
           "sandbox": {"rejected_fraction": rejected, "n_unsafe": len(unsafe), "net_isolation": sandbox.net_isolation(),
                       "per_script": {k: v.stage for k, v in unsafe.items()}},
           "d034": {"criteria": crit, "verdict": "PASS" if passed else "FAIL",
                    "action": "promote to Studio's default for engineering prompts" if passed else
                              "ship as opt-in (BERKELIUM_CAD_BACKEND=cadquery), not the default"},
           "items_file": str(items_file), "items_sha256": sha256_file(items_file)}
    rd = Path(a.results)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "E-C2.json").write_text(json.dumps(res, indent=1, default=str))
    (rd / "E-C2.report.md").write_text(report_md(res))
    print(f"D-034: {res['d034']['verdict']} -> {rd / 'E-C2.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
