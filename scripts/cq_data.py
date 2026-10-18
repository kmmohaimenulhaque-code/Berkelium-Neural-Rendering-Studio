#!/usr/bin/env python3
"""Build the verified text → CadQuery dataset ``berkelium-cq-v1`` (runs on the MI300X host, all CPU cores).

  python scripts/cq_data.py build --out ~/berkelium_run/data/cq_v1 --gen-hours 2
  python scripts/cq_data.py build --out /tmp/cq_smoke --smoke          # tiny, offline-capable sanity run

Sources (revisions pinned in cadquery/pins.py; each download has a 20 min budget, a source that misses it is
skipped and recorded in the manifest):

1. **text2cadquery**: every Text-to-CadQuery pair (arXiv 2505.06507, ~170k). Export/import lines are stripped,
   ``result = <final part>`` is appended when missing. Its own train/val/test split is kept (test = its 5 %).
2. **families**: Berkelium parametric families → Geometry IR → CadQuery (cadquery/transpile.py), 20k rows. A row
   is kept only if the sandboxed CadQuery solid matches the core's OCCT solid (volume rel 1e-6, bbox 1e-3 mm).
   ``pipe_elbow``, ``chair``, ``knob``, ``hinge_leaf`` go to ``heldout`` only; 5 % of the rest to ``test``.
3. **repair**: 10k rows built from verified *train* rows: the script is broken (undefined name, misspelt method,
   missing ``result``, syntax error, missing import), the sandbox error is recorded, and the target is the fix.
4. **cad_recode**: CAD-Recode v1.5 programs (arXiv 2412.14042, ~1M, CC-BY-NC-4.0) verified in the sandbox, with
   request texts written by the base model through vLLM (HF transformers fallback) inside ``--gen-hours``.
   Rows without a generated text are not used. 5 % by hash to ``test``.

Every script is run through cadquery/sandbox.py (static whitelist + isolated child); rejected rows are kept with
their reason in ``rejections.jsonl.gz``. Train rows whose text or code also occurs in test/heldout are dropped.
The manifest records revisions, SHA-256 of every input and output file, counts and rejection reasons."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "cadquery"), str(ROOT / "services" / "neural")]

import cqformat  # noqa: E402
import pins  # noqa: E402
import sandbox  # noqa: E402

DOWNLOAD_BUDGET_S = 20 * 60
T2CQ_ASSUME = "the numbers in the request are used as given; the script treats 1 unit as 1 mm"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hkey(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def norm_code(c: str) -> str:
    return "\n".join(x.split("#", 1)[0].rstrip() for x in c.splitlines() if x.split("#", 1)[0].strip())


def bucket(key: str) -> int:
    return int(hkey(key)[:8], 16) % 100


def first_sentence(t: str, n: int = 160) -> str:
    s = re.split(r"(?<=[.!?])\s", t.strip(), maxsplit=1)[0]
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


# ------------------------------------------------------------------------------------------------ downloads
def download_hf_files(spec: dict, dest: Path, budget: float) -> dict:
    """Pinned files via huggingface_hub in a child process with a hard time budget."""
    code = ("import sys, json; from huggingface_hub import hf_hub_download as d; a = json.loads(sys.argv[1]); "
            "[print(d(a['repo'], f, repo_type=a['type'], revision=a['rev'], local_dir=a['dest'])) for f in a['files']]")
    arg = json.dumps({"repo": spec["hf_repo"], "type": spec["repo_type"], "rev": spec["revision"],
                      "files": spec["files"], "dest": str(dest)})
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", code, arg], capture_output=True, text=True, timeout=budget)
    except subprocess.TimeoutExpired:
        return {"status": "skipped", "reason": f"download exceeded {budget / 60:.0f} min", "seconds": budget}
    if p.returncode != 0:
        return {"status": "skipped", "reason": "download failed: " + p.stderr.strip()[-500:],
                "seconds": round(time.time() - t0, 1)}
    files = {f: sha256_file(dest / f) for f in spec["files"]}
    return {"status": "ok", "seconds": round(time.time() - t0, 1), "files_sha256": files}


def download_git(spec: dict, dest: Path, budget: float) -> dict:
    """A repo of ~1M small files: one shallow git clone (a single pack) is far faster than per-file HTTP."""
    t0 = time.time()
    url = f"https://huggingface.co/{'datasets/' if spec['repo_type'] == 'dataset' else ''}{spec['hf_repo']}"
    if (dest / ".git").exists():
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == spec["revision"]:
            return {"status": "ok", "seconds": 0.0, "git_head": head, "cached": True}
        shutil.rmtree(dest)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    try:
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(dest)], check=True, env=env,
                       timeout=budget, capture_output=True)
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != spec["revision"]:
            left = max(60.0, budget - (time.time() - t0))
            subprocess.run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", spec["revision"]],
                           check=True, env=env, timeout=left, capture_output=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "-q", spec["revision"]], check=True, env=env,
                           capture_output=True)
            head = spec["revision"]
    except subprocess.TimeoutExpired:
        shutil.rmtree(dest, ignore_errors=True)
        return {"status": "skipped", "reason": f"download exceeded {budget / 60:.0f} min", "seconds": budget}
    except subprocess.CalledProcessError as e:
        shutil.rmtree(dest, ignore_errors=True)
        return {"status": "skipped", "reason": "git clone failed: " + (e.stderr or b"").decode()[-500:],
                "seconds": round(time.time() - t0, 1)}
    return {"status": "ok", "seconds": round(time.time() - t0, 1), "git_head": head}


# ------------------------------------------------------------------------------------------------ cleaning
_EXPORT = re.compile(r"^\s*(from\s+cadquery\s+import\s+exporters\b.*|from\s+cadquery\.vis\s+import.*|"
                     r".*\bexporters\s*\.\s*export\s*\(.*|.*\bshow_object\s*\(.*|.*\bdisplay\s*\(.*|"
                     r"show\s*\(.*)$")


def clean_script(code: str, final_hint: str | None = None) -> str:
    """Drop exporter/display lines and make sure the script ends with ``result = ...``."""
    lines = [x for x in code.replace("\r\n", "\n").split("\n") if not _EXPORT.match(x)]
    code = "\n".join(lines).strip() + "\n"
    if re.search(r"^result\s*=", code, re.M):
        return code
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    names = [t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]
    pick = final_hint if final_hint in names else next((n for n in reversed(names)
                                                        if n in ("assembly", "final_result", "final", "r")), None)
    pick = pick or (names[-1] if names else None)
    return code + f"\nresult = {pick}\n" if pick else code


# ------------------------------------------------------------------------------------------------ sources
def rows_text2cadquery(src_dir: Path, limit: int | None) -> list[dict]:
    rows = []
    for split, fn in (("train", "data_train.jsonl"), ("val", "data_val.jsonl"), ("test", "data_test.jsonl")):
        p = src_dir / "data" / fn
        with open(p) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                d = json.loads(line)
                code = clean_script(d["output"])
                header = {"summary": first_sentence(d["input"]), "assumptions": [T2CQ_ASSUME], "open_questions": []}
                rows.append({"id": f"t2cq-{split}-{i}", "source": "text2cadquery", "split": split, "kind": "gen",
                             "text": d["input"].strip(), "code": code, "header": header})
    return rows


def _family_row(args):
    fam, seed, vague = args
    from berkelium_neural.cad import families as F
    import transpile
    r = F.sample(fam, seed, vague)
    if r is None:
        return None, {"family": fam, "seed": seed, "reason": "sampler rejected the parameters"}
    text, prop = r
    try:
        from berkelium.geometry.occt import OCCTBackend
        from berkelium.pipeline import run
        rec = run(prop, realize=False).record
        g = rec.realized[0].geometry
        ops = [o.model_dump() for o in g.ops]
        code = transpile.transpile(prop, resolved_ops=ops)
        occ = OCCTBackend()
        m = occ.measure(occ.execute(g).bodies["body"])
    except Exception as e:  # noqa: BLE001
        return None, {"family": fam, "seed": seed, "reason": f"core/transpile: {type(e).__name__}: {e}"[:300]}
    it = prop["intent"]
    header = {"summary": it.get("summary", fam), "assumptions": it.get("assumptions", []),
              "open_questions": it.get("open_questions", [])}
    heldout = fam in pins.HELDOUT_FAMILIES
    key = f"{fam}:{seed}:{int(vague)}"
    split = "heldout" if heldout else ("test" if bucket(key) < 5 else "train")
    return {"id": f"fam-{fam}-{seed}-{int(vague)}", "source": "families", "split": split, "kind": "gen",
            "family": fam, "text": text, "code": code, "header": header,
            "core_volume": m.volume_mm3, "core_bbox": list(m.bbox_min) + list(m.bbox_max)}, None


def rows_families(n: int, workers: int) -> tuple[list[dict], list[dict]]:
    from berkelium_neural.cad import families as F
    fams = list(F.FAMILIES)
    per = max(1, -(-n // len(fams)))
    jobs = [(f, s, s % 10 < 3) for f in fams for s in range(per)]
    rows, rej = [], []
    with Pool(workers) as pool:
        for r, bad in pool.imap_unordered(_family_row, jobs, chunksize=8):
            (rows.append(r) if r else rej.append({"source": "families", **bad}))
    rows.sort(key=lambda r: r["id"])
    return rows, rej


def rows_cad_recode(src_dir: Path, max_rows: int, seed: int = 0) -> list[dict]:
    files = sorted(str(p.relative_to(src_dir)) for p in src_dir.glob("*/batch_*/*.py"))
    files += sorted(str(p.relative_to(src_dir)) for p in (src_dir / "val").glob("*.py"))
    random.Random(seed).shuffle(files)
    rows = []
    for rel in files[:max_rows]:
        code = clean_script((src_dir / rel).read_text(), final_hint="r")
        split = "test" if bucket(rel) < 5 else "train"
        rows.append({"id": "rec-" + rel.replace("/", "-")[:-3], "source": "cad_recode", "split": split,
                     "kind": "gen", "text": None, "code": code, "header": None})
    return rows


# ------------------------------------------------------------------------------------------------ repair rows
def break_script(code: str, rng: random.Random) -> tuple[str, str] | None:
    """One deterministic corruption of a working script, or None if this script offers none."""
    lines = code.split("\n")
    kinds = ["name", "method", "result", "syntax", "import"]
    rng.shuffle(kinds)
    for k in kinds:
        if k == "result":
            idx = [i for i, x in enumerate(lines) if re.match(r"^result\s*=", x)]
            if idx:
                return "\n".join(lines[: idx[-1]] + lines[idx[-1] + 1:]), k
        elif k == "import" and "import cadquery as cq" in code:
            return code.replace("import cadquery as cq\n", "", 1), k
        elif k == "method":
            ms = [m for m in re.finditer(r"\.(extrude|circle|rect|box|cylinder|union|cut|hole|translate|rotate|"
                                         r"revolve|polyline|lineTo|moveTo|workplane|fuse)\(", code)]
            if ms:
                m = ms[rng.randrange(len(ms))]
                bad = m.group(1) + rng.choice(["e", "s", "_", "x"])
                return code[: m.start(1)] + bad + code[m.end(1):], k
        elif k == "name":
            import ast
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            assigned = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                        if isinstance(t, ast.Name)} - {"result"}
            uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    and n.id in assigned]
            if uses:
                u = uses[rng.randrange(len(uses))]
                ln = lines[u.lineno - 1]
                lines2 = list(lines)
                lines2[u.lineno - 1] = ln[: u.col_offset] + u.id + "_value" + ln[u.col_offset + len(u.id):]
                return "\n".join(lines2), k
        elif k == "syntax":
            idx = [i for i, x in enumerate(lines) if x.rstrip().endswith(")") and "(" in x]
            if idx:
                i = idx[rng.randrange(len(idx))]
                lines2 = list(lines)
                lines2[i] = lines2[i].rstrip()[:-1]
                return "\n".join(lines2), k
    return None


# ------------------------------------------------------------------------------------------------ text generation
GEN_SYSTEM = ("You write the request an engineer would type into a CAD assistant to get exactly the part that a "
              "CadQuery script builds. Describe the shape, its features and the key dimensions in mm, in one to "
              "three plain sentences. Never mention CadQuery, code, workplanes or variable names. Reply with the "
              "request only.")


def generate_texts(codes: list[str], model: str, revision: str, hours: float, tp: int = 1) -> list[str | None]:
    """Request texts for scripts, written by the base model, until the time budget runs out."""
    deadline = time.time() + hours * 3600
    out: list[str | None] = [None] * len(codes)
    msgs = [[{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": c[:6000]}] for c in codes]
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model, revision=revision, tensor_parallel_size=tp, dtype="bfloat16", max_model_len=8192,
                  gpu_memory_utilization=0.90, seed=0)
        sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200, seed=0)
        for i in range(0, len(msgs), 2048):
            if time.time() > deadline:
                break
            res = llm.chat(msgs[i:i + 2048], sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
            for j, r in enumerate(res):
                out[i + j] = r.outputs[0].text.strip() or None
        backend = "vllm"
    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, revision=revision, padding_side="left")
        m = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16, device_map="auto").eval()
        for i in range(0, len(msgs), 32):
            if time.time() > deadline:
                break
            ps = [tok.apply_chat_template(x, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                  for x in msgs[i:i + 32]]
            enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
            with torch.no_grad():
                g = m.generate(**enc, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
            for j, row in enumerate(g):
                out[i + j] = tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() or None
        backend = "transformers"
    generate_texts.backend = backend  # type: ignore[attr-defined]
    return out


# ------------------------------------------------------------------------------------------------ build
def verify(rows: list[dict], runner: sandbox.BatchRunner, rej: list[dict], check_core: bool = False) -> list[dict]:
    keep = []
    for row, res in zip(rows, runner.map([r["code"] for r in rows])):
        if not res.ok:
            rej.append({"source": row["source"], "id": row["id"], "reason": f"sandbox {res.stage}: {res.error[:300]}"})
            continue
        if check_core:
            cv, cb = row["core_volume"], row["core_bbox"]
            if abs(res.volume - cv) > 1e-6 * max(1.0, abs(cv)) or max(abs(a - b) for a, b in zip(res.bbox, cb)) > 1e-3:
                rej.append({"source": row["source"], "id": row["id"],
                            "reason": f"transpile mismatch: volume {res.volume:.6g} vs core {cv:.6g}"})
                continue
        row["ref_volume"], row["ref_bbox"] = res.volume, res.bbox
        keep.append(row)
    return keep


def finalize(row: dict) -> dict:
    row["completion"] = cqformat.render(row["header"], row["code"])
    return row


def build(a) -> int:
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    workers = a.workers or os.cpu_count() or 1
    t_start = time.time()
    man: dict = {"dataset": "berkelium-cq-v1", "format": cqformat.FORMAT_VERSION, "created_unix": int(t_start),
                 "pins": {"core": pins.CORE, "sources": pins.SOURCES}, "workers": workers, "sources": {},
                 "args": vars(a), "sandbox": {"net_isolation": sandbox.net_isolation(), "timeout_s": sandbox.TIMEOUT_S,
                                              "mem_bytes": sandbox.MEM_BYTES}}
    try:
        import cadquery
        man["cadquery_version"] = cadquery.__version__
    except ImportError:
        man["cadquery_version"] = None
    rows: list[dict] = []
    rej: list[dict] = []
    want = set(a.sources.split(","))
    runner = sandbox.BatchRunner(jobs=workers)
    try:
        if "text2cadquery" in want:
            spec = pins.SOURCES["text2cadquery"]
            d = cache / "text2cadquery"
            info = ({"status": "ok", "local": a.t2cq_dir} if a.t2cq_dir else
                    download_hf_files(spec, d, a.download_budget))
            man["sources"]["text2cadquery"] = info
            if info["status"] == "ok":
                t0 = time.time()
                cand = rows_text2cadquery(Path(a.t2cq_dir) if a.t2cq_dir else d, a.t2cq_limit)
                kept = verify(cand, runner, rej)
                rows += kept
                info.update(candidates=len(cand), verified=len(kept), verify_seconds=round(time.time() - t0, 1))
            print("text2cadquery:", json.dumps(info)[:400], flush=True)
        if "families" in want:
            t0 = time.time()
            cand, frej = rows_families(a.families_n, workers)
            rej += frej
            kept = verify(cand, runner, rej, check_core=True)
            rows += kept
            man["sources"]["families"] = {"status": "ok", "candidates": len(cand) + len(frej), "verified": len(kept),
                                          "heldout_families": list(pins.HELDOUT_FAMILIES),
                                          "seconds": round(time.time() - t0, 1)}
            print("families:", man["sources"]["families"], flush=True)
        if "cad_recode" in want:
            spec = pins.SOURCES["cad_recode"]
            d = cache / "cad_recode"
            info = {"status": "ok", "local": a.recode_dir} if a.recode_dir else download_git(spec, d, a.download_budget)
            man["sources"]["cad_recode"] = info
            if info["status"] == "ok":
                t0 = time.time()
                cand = rows_cad_recode(Path(a.recode_dir) if a.recode_dir else d, a.recode_max)
                kept = verify(cand, runner, rej)
                info.update(candidates=len(cand), verified=len(kept), verify_seconds=round(time.time() - t0, 1))
                if a.gen_hours > 0 and kept:
                    t1 = time.time()
                    texts = generate_texts([r["code"] for r in kept], a.gen_model, pins.MODELS[a.gen_model]["revision"]
                                           if a.gen_model in pins.MODELS else "main", a.gen_hours, a.gen_tp)
                    with_text = []
                    for r, t in zip(kept, texts):
                        if t:
                            r["text"] = t
                            r["header"] = {"summary": first_sentence(t), "assumptions": [], "open_questions": []}
                            with_text.append(r)
                    info.update(text_backend=getattr(generate_texts, "backend", None), text_model=a.gen_model,
                                with_text=len(with_text), gen_seconds=round(time.time() - t1, 1))
                    rows += with_text
                else:
                    info.update(with_text=0, note="--gen-hours 0: no texts generated, source not used")
            print("cad_recode:", json.dumps(info)[:400], flush=True)

        # ---- dedup train against every evaluation split
        held = [r for r in rows if r["split"] in ("test", "heldout")]
        bad_t = {hkey(norm_text(r["text"])) for r in held}
        bad_c = {hkey(norm_code(r["code"])) for r in held}
        n0 = len(rows)
        dup = Counter()
        kept_rows = []
        for r in rows:
            if r["split"] in ("train", "val") and (hkey(norm_text(r["text"])) in bad_t or hkey(norm_code(r["code"])) in bad_c):
                dup[r["source"]] += 1
                continue
            kept_rows.append(r)
        rows = kept_rows
        man["dedup_removed"] = dict(dup)
        print(f"dedup: removed {n0 - len(rows)} train/val rows that overlap test/heldout", flush=True)

        # ---- repair rows from verified train rows
        if "repair" in want and a.repair_n > 0:
            rng = random.Random(1234)
            pool = [r for r in rows if r["split"] == "train"]
            rng.shuffle(pool)
            cand = []
            for r in pool[: a.repair_n * 2]:
                b = break_script(r["code"], rng)
                if b:
                    cand.append((r, b[0], b[1]))
            reps = []
            for (r, broken, kind), res in zip(cand, runner.map([c[1] for c in cand])):
                if res.ok:
                    rej.append({"source": "repair", "id": r["id"], "reason": f"corruption {kind} still ran"})
                    continue
                err = f"{res.stage}: {res.error}"
                bad_completion = cqformat.render(r["header"], broken)
                reps.append({"id": f"rep-{r['id']}-{kind}", "source": "repair", "split": "train", "kind": "repair",
                             "corruption": kind, "text": r["text"], "code": r["code"], "header": r["header"],
                             "broken_completion": bad_completion, "error": err[:1500], "parent": r["id"]})
                if len(reps) >= a.repair_n:
                    break
            rows += reps
            man["sources"]["repair"] = {"status": "ok", "verified": len(reps),
                                        "by_corruption": dict(Counter(r["corruption"] for r in reps))}
            print("repair:", man["sources"]["repair"], flush=True)
    finally:
        runner.close()

    # ---- write
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(finalize(r))
    files = {}
    for split, rs in sorted(by_split.items()):
        p = out / f"{split}.jsonl.gz"
        with gzip.open(p, "wt") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[p.name] = {"rows": len(rs), "sha256": sha256_file(p)}
    p = out / "rejections.jsonl.gz"
    with gzip.open(p, "wt") as f:
        for r in rej:
            f.write(json.dumps(r) + "\n")
    files[p.name] = {"rows": len(rej), "sha256": sha256_file(p)}
    reasons = Counter(re.sub(r"[:(].*", "", r["reason"]).strip() for r in rej)
    man.update(files=files, counts={s: dict(Counter(r["source"] for r in rs)) for s, rs in by_split.items()},
               rejections={"total": len(rej), "by_reason": dict(reasons.most_common(30)),
                           "by_source": dict(Counter(r["source"] for r in rej))},
               seconds=round(time.time() - t_start, 1))
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({"out": str(out), "files": files, "rejections": len(rej)}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="~/berkelium_run/data/cq_v1")
    b.add_argument("--cache", default="~/berkelium_run/cache")
    b.add_argument("--workers", type=int, default=0)
    b.add_argument("--sources", default="text2cadquery,families,repair,cad_recode")
    b.add_argument("--families-n", type=int, default=20000)
    b.add_argument("--repair-n", type=int, default=10000)
    b.add_argument("--recode-max", type=int, default=1_000_000)
    b.add_argument("--t2cq-limit", type=int, default=None, help="rows per split (default: all)")
    b.add_argument("--t2cq-dir", default=None, help="use a local copy instead of downloading")
    b.add_argument("--recode-dir", default=None, help="use a local copy instead of downloading")
    b.add_argument("--gen-hours", type=float, default=2.0)
    b.add_argument("--gen-model", default="Qwen/Qwen3-32B")
    b.add_argument("--gen-tp", type=int, default=1)
    b.add_argument("--download-budget", type=float, default=DOWNLOAD_BUDGET_S)
    b.add_argument("--smoke", action="store_true", help="tiny sizes: 40 family rows, 10 repair rows, 20 t2cq/split")
    a = ap.parse_args()
    if a.smoke:
        a.families_n, a.repair_n, a.recode_max = min(a.families_n, 40), min(a.repair_n, 10), min(a.recode_max, 20)
        a.t2cq_limit = a.t2cq_limit or 20
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
