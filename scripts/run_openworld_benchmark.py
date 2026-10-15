#!/usr/bin/env python3
"""E-T3: open-world text-to-3D benchmark on CPU (B1-B4). Generation and scoring are separate, resumable phases.

  python scripts/run_openworld_benchmark.py plan       # case list, generation counts, budget check (no model calls)
  python scripts/run_openworld_benchmark.py generate   # every planned generation unit not yet in units.jsonl
  python scripts/run_openworld_benchmark.py score      # metrics from saved meshes -> E-T3_openworld_cpu.json
  python scripts/run_openworld_benchmark.py report     # -> E-T3_openworld_cpu.report.md (numbers from the JSON)

Config: benchmarks/configs/text3d_openworld_cpu_v1.json (+ the frozen prompt list it names). A generation unit is
(generator, prompt, seed); benchmark rows reference units, so identical requests are generated once (auto reuses the
unit of the generator it routes to; the planner reuses part units). Shap-E units run one per subprocess
(scripts/openworld_generate.py) so each starts from a clean heap and a watchdog kill is recorded, not fatal. Units are
started in the config's priority order until the Shap-E budget is spent; every unstarted unit is listed as omitted.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from _common import ROOT, provenance

CFG = ROOT / "benchmarks" / "configs" / "text3d_openworld_cpu_v1.json"
NATIVE, SHAP_E, AUTO = "berkelium-textshape", "shap-e-text300m", "auto"


def load_cfg():
    cfg = json.loads(CFG.read_text())
    prompts = json.loads((ROOT / cfg["prompts"]).read_text())
    return cfg, prompts


def slug(prompt: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:60]


def unit_key(gen: str, prompt: str, seed: int) -> str:
    return f"{gen}|{prompt}|{seed}"


def unit_dir(out: Path, gen: str, prompt: str, seed: int) -> Path:
    h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    return out / "units" / gen / f"{slug(prompt)}__{h}__s{seed}"


def records(prompts, key):
    from berkelium_neural.text3d.dataset import load_manifest
    root = ROOT / "data" / "procedural_shapes_v1"
    by_id = {r.asset_id: r for r in load_manifest(root, {"test"})}
    return [by_id[a] for a in prompts[key]["asset_ids"]], root


def all_benchmark_prompts(prompts):
    b3, _ = records(prompts, "B3_retention")
    b4, _ = records(prompts, "B4_composition")
    return ([("B1", o["prompt"]) for o in prompts["B1_open_world_gt"]["objects"]] +
            [("B2", p["prompt"]) for p in prompts["B2_gt_free"]["prompts"]] +
            [("B3", r.caption) for r in b3] + [("B4", r.caption) for r in b4])


def plan_units(cfg, prompts):
    """-> (ordered list of units {gen, prompt, seed, group}, routing decisions per benchmark prompt)."""
    from berkelium_neural.text3d import planner
    from berkelium_neural.text3d.generators import route
    from berkelium_neural.text3d.model import build_vocab
    vocab, seed = build_vocab(), cfg["seed"]
    routing = {p: route(p, vocab) for _, p in all_benchmark_prompts(prompts)}
    b4, _ = records(prompts, "B4_composition")
    parts = []
    for r in b4:
        pl, _ = planner.parse(r.caption)
        parts += [pl.a, pl.b] if pl else []
    groups = {
        "native": [p for _, p in all_benchmark_prompts(prompts)] + parts,
        "B1": [o["prompt"] for o in prompts["B1_open_world_gt"]["objects"]],
        "B4_single": [r.caption for r in b4],
        "B2": [p["prompt"] for p in prompts["B2_gt_free"]["prompts"]],
        "B4_planner": parts,
        "B3_routed": [p for g, p in all_benchmark_prompts(prompts) if g == "B3" and routing[p]["generator"] == SHAP_E],
    }
    units, seen = [], set()
    for group in cfg["priority"]:
        gen = NATIVE if group == "native" else SHAP_E
        for p in groups[group]:
            k = unit_key(gen, p, seed)
            if k not in seen:
                seen.add(k)
                units.append({"gen": gen, "prompt": p, "seed": seed, "group": group})
    return units, routing


def read_units(out: Path) -> dict:
    p = out / "units.jsonl"
    rows = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["key"]] = r
    return rows


def append_unit(out: Path, row: dict):
    with open(out / "units.jsonl", "a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def run_native(u, out: Path, cfg) -> dict:
    from berkelium_neural.text3d.generate import mesh_to_glb
    from berkelium_neural.text3d.generators import NativeTextShape
    g = NativeTextShape(adapter=cfg["native"]["adapter"], resolution=cfg["native"]["resolution"])
    t0 = time.time()
    o = g.generate(u["prompt"], u["seed"])
    d = unit_dir(out, u["gen"], u["prompt"], u["seed"])
    d.mkdir(parents=True, exist_ok=True)
    if o["vertices"] is not None:
        np.savez_compressed(d / "mesh.npz", v=o["vertices"], f=o["faces"],
                            col=o["colors"] if o["colors"] is not None else np.zeros((0, 3), np.float32))
        (d / "mesh.glb").write_bytes(mesh_to_glb(o["vertices"], o["faces"], o["colors"]))
    rep = {**o["report"], "identity": g.identity, "wall_s": round(time.time() - t0, 2)}
    (d / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    return {"status": "ok" if rep.get("valid") else "invalid_mesh", "dir": str(d.relative_to(ROOT)),
            "valid": bool(rep.get("valid")), "seconds": rep.get("seconds")}


def run_shap_e(u, out: Path, cfg) -> dict:
    s = cfg["shap_e"]
    d = unit_dir(out, u["gen"], u["prompt"], u["seed"])
    d.parent.mkdir(parents=True, exist_ok=True)
    log = d.parent / (d.name + ".log")
    cmd = [sys.executable, str(ROOT / "scripts" / "openworld_generate.py"), u["prompt"], "--steps", str(s["steps"]),
           "--guidance", str(s["guidance"]), "--seed", str(u["seed"]), "--threads", str(s["threads"]),
           "--out", str(d.parent), "--dir-name", d.name]
    t0 = time.time()
    with open(log, "w") as fh:
        try:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=s["timeout_s"], check=False).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    wall = round(time.time() - t0, 1)
    rep = json.loads((d / "report.json").read_text()) if (d / "report.json").exists() else {}
    if rc == 0 or (rc == 1 and rep.get("status") == "invalid_mesh"):
        status = rep.get("status", "invalid_mesh")
    elif rc == 86:
        status = "aborted_memory"
    else:
        status = f"failed(rc={rc})"
    return {"status": status, "dir": str(d.relative_to(ROOT)), "valid": bool(rep.get("valid")), "rc": rc,
            "wall_s": wall, "seconds": rep.get("seconds"), "sample_s": rep.get("sample_s"),
            "decode_s": rep.get("decode_s"), "peak_rss_mib": (rep.get("memory_mib") or {}).get("peak_VmRSS"),
            "log": str(log.relative_to(ROOT))}


def cmd_plan(cfg, prompts, out):
    units, routing = plan_units(cfg, prompts)
    done = read_units(out)
    n_shap = sum(u["gen"] == SHAP_E for u in units)
    budget = cfg["budget"]["max_shap_e_generations"]
    print(f"{len(units)} units: {len(units) - n_shap} native, {n_shap} shap-e (budget {budget}); "
          f"{sum(unit_key(u['gen'], u['prompt'], u['seed']) in done for u in units)} done")
    for g in cfg["priority"]:
        sel = [u for u in units if u["group"] == g]
        print(f"  {g:12s} {len(sel):3d} units ({sel[0]['gen'] if sel else '-'})")
    print("routing:", {k: sum(r['generator'] == k for r in routing.values()) for k in (NATIVE, SHAP_E)})
    return units


def cmd_generate(cfg, prompts, out):
    out.mkdir(parents=True, exist_ok=True)
    units, _ = plan_units(cfg, prompts)
    done = read_units(out)
    budget = cfg["budget"]["max_shap_e_generations"]
    deadline = time.time() + cfg["budget"]["wall_hours_shap_e"] * 3600
    used = sum(1 for r in done.values() if r["gen"] == SHAP_E)
    for u in units:
        k = unit_key(u["gen"], u["prompt"], u["seed"])
        if k in done:
            continue
        if u["gen"] == SHAP_E:
            if used >= budget or time.time() > deadline:
                print(f"budget reached ({used}/{budget} shap-e units or wall deadline); stopping before {k}")
                break
            used += 1
        t0 = time.time()
        res = run_native(u, out, cfg) if u["gen"] == NATIVE else run_shap_e(u, out, cfg)
        row = {"key": k, **u, **res, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        append_unit(out, row)
        print(f"[{u['group']}] {u['gen']} {u['prompt']!r}: {res['status']} ({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------------------------------- scoring
def load_mesh(row):
    if row is None or not row.get("valid"):
        return None
    z = np.load(ROOT / row["dir"] / "mesh.npz")
    rep = json.loads((ROOT / row["dir"] / "report.json").read_text())
    return {"vertices": z["v"], "faces": z["f"], "colors": z["col"] if z["col"].size else None, "report": rep}


def mesh_or_empty(row):
    m = load_mesh(row)
    if m is not None:
        return m
    rep = json.loads((ROOT / row["dir"] / "report.json").read_text()) if row and (ROOT / row["dir"] /
                                                                                  "report.json").exists() else {}
    return {"vertices": None, "faces": None, "colors": None,
            "report": {"valid": False, "seconds": rep.get("seconds"), "reason": row["status"] if row else "omitted"}}


class _Fixed:
    """evaluate.py's generator interface over an already generated (saved) output."""

    def __init__(self, out, name):
        self.out, self.name = out, name

    def generate(self, prompt):
        rep = dict(self.out["report"])
        rep.setdefault("seconds", None)
        return {**self.out, "report": rep}


def clip_evaluator(cfg):
    """OpenAI CLIP ViT-B/32 (NOT the ViT-L/14 that conditions Shap-E), loaded from its sha256-pinned TorchScript
    archive with the restricted reader (no TorchScript execution)."""
    from berkelium_neural.text3d.shap_e_backend import (
        read_torchscript_tensors,
        sha256_file,
    )
    from clip.clip import _transform
    from clip.model import build_model
    p = ROOT / cfg["evaluator"]["path"]
    if sha256_file(p) != cfg["evaluator"]["sha256"]:
        raise RuntimeError(f"{p} does not match the pinned evaluator sha256")
    sd, _ = read_torchscript_tensors(p, lambda n: True)
    for k in ("input_resolution", "context_length", "vocab_size"):
        sd.pop(k, None)
    model = build_model(sd).float().eval()
    return model, _transform(model.visual.input_resolution)


def embed_texts(model, texts):
    import clip
    import torch
    with torch.no_grad():
        e = model.encode_text(clip.tokenize(texts, truncate=True)).float()
    return (e / e.norm(dim=-1, keepdim=True)).numpy()


def embed_views(model, preprocess, images):
    import torch
    from PIL import Image
    x = torch.stack([preprocess(Image.fromarray((np.clip(im, 0, 1) * 255).round().astype(np.uint8))) for im in images])
    with torch.no_grad():
        e = model.encode_image(x).float()
    e = e / e.norm(dim=-1, keepdim=True)
    m = e.mean(0)
    return (m / m.norm()).numpy()


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def cmd_score(cfg, prompts, out):
    from berkelium_neural.fitting.real_object import (
        contact_sheet,
        load_scan,
        render_scan,
        save_png,
    )
    from berkelium_neural.text3d import planner
    from berkelium_neural.text3d.evaluate import evaluate_record
    from berkelium_neural.text3d.openworld_eval import (
        align_and_score,
        r_precision,
        touch_components,
        view_images,
    )
    units_planned, routing = plan_units(cfg, prompts)
    done = read_units(out)
    seed = cfg["seed"]
    U = lambda gen, p: done.get(unit_key(gen, p, seed))
    omitted = [unit_key(u["gen"], u["prompt"], u["seed"]) for u in units_planned
               if unit_key(u["gen"], u["prompt"], u["seed"]) not in done]

    def routed(p):
        return routing[p]["generator"]

    result = {"experiment": "E-T3", "protocol": "berkelium-text3d-eval/1 + E-T3 extensions (openworld_eval.py)",
              "config": cfg, "prompts_file": cfg["prompts"],
              "prompts_sha256": hashlib.sha256((ROOT / cfg["prompts"]).read_bytes()).hexdigest(),
              "training_performed": "none", "device": "cpu", "environment": provenance(),
              "omitted_units": omitted, "completeness": "full" if not omitted else "partial"}

    # ---- B1: open-world prompts with real-object ground truth (generous alignment)
    b1 = []
    gt_cache = {}
    for o in prompts["B1_open_world_gt"]["objects"]:
        z = ROOT / "data" / "gso" / o["name"] / f"{o['name']}.zip"
        scan = load_scan(z, tex_size=256)
        if scan.source["zip_sha256"] != o["zip_sha256"]:
            raise RuntimeError(f"{o['name']}: scan zip changed since the prompt list was frozen")
        gt_cache[o["prompt"]] = scan
        row = {"object": o["name"], "category": o["category"], "prompt": o["prompt"], "route": routing[o["prompt"]]}
        for cand, gen in ((NATIVE, NATIVE), (SHAP_E, SHAP_E), (AUTO, routed(o["prompt"]))):
            u = U(gen, o["prompt"])
            m = load_mesh(u)
            row[cand] = {"status": u["status"] if u else "omitted", "valid": bool(m), "generator": gen,
                         "seconds": u.get("seconds") if u else None,
                         **({k: v for k, v in align_and_score(m["vertices"], m["faces"], scan.v, scan.f).items()}
                            if m else {"chamfer_l1": None, "fscore@0.05": 0.0 if u else None})}
        b1.append(row)
    agg1 = {}
    for cand in (NATIVE, SHAP_E, AUTO):
        rows = [r[cand] for r in b1 if r[cand]["status"] != "omitted"]
        agg1[cand] = {"n": len(rows), "valid_rate": mean([float(r["valid"]) for r in rows]),
                      "chamfer_l1_mean": mean([r.get("chamfer_l1") for r in rows]),
                      "fscore@0.05_mean": mean([r.get("fscore@0.05") for r in rows]),
                      "fscore@0.02_mean": mean([r.get("fscore@0.02") for r in rows])}
    both = [r for r in b1 if r[NATIVE].get("chamfer_l1") is not None and r[SHAP_E].get("chamfer_l1") is not None]
    agg1["shap_e_better_chamfer"] = f"{sum(r[SHAP_E]['chamfer_l1'] < r[NATIVE]['chamfer_l1'] for r in both)}/{len(both)}"
    result["B1"] = {"rows": b1, "aggregate": agg1, "alignment": "unit extent + best of 24 axis rotations + ICP "
                    "(same for every generator; generous: forgives pose, frame and scale)"}

    # ---- B2: semantic relevance (evaluator CLIP ViT-B/32), B1 + GT-free prompts, all benchmark prompts as candidates
    model, preprocess = clip_evaluator(cfg)
    cand_prompts = [p for _, p in all_benchmark_prompts(prompts)]
    T = embed_texts(model, cand_prompts)
    sem_prompts = [o["prompt"] for o in prompts["B1_open_world_gt"]["objects"]] + \
        [p["prompt"] for p in prompts["B2_gt_free"]["prompts"]]
    b2 = {"evaluator": cfg["evaluator"], "candidates": len(cand_prompts), "views": 4, "rows": []}
    sheets = []
    for cand in ("gt_scan", NATIVE, SHAP_E, AUTO):
        embs, truth, rows = [], [], []
        for p in sem_prompts:
            if cand == "gt_scan":
                if p not in gt_cache:
                    continue
                from berkelium_neural.camera import orbit_cameras
                cams = orbit_cameras(4, 3.2, 25.0, 224, 224, phase_deg=30.0)
                ims = []
                for c in cams:
                    rgb, a = render_scan(gt_cache[p], c)[:2]
                    ims.append(rgb + (1 - a[..., None]))        # premultiplied on black -> white background
                gen, st = "gt", "ok"
            else:
                gen = cand if cand != AUTO else routed(p)
                u = U(gen, p)
                if u is None:
                    continue
                m = mesh_or_empty(u)
                ims, st = view_images(m["vertices"], m["faces"], m["colors"]), u["status"]
            embs.append(embed_views(model, preprocess, ims))
            truth.append(cand_prompts.index(p))
            rows.append({"prompt": p, "generator": gen, "status": st})
            if cand in (SHAP_E, NATIVE, "gt_scan") and p in sem_prompts[:4]:
                sheets.append((cand, p, ims))
        if not embs:
            continue
        rp = r_precision(np.stack(embs), T, truth)
        for r, k in zip(rows, rp["ranks"], strict=True):
            r["rank"] = k
        b2["rows"].append({"candidate": cand, "n": len(rows), "r_at_1": rp["r_at_1"], "r_at_5": rp["r_at_5"],
                           "per_prompt": rows})
    b2["residual_bias"] = ("evaluator (ViT-B/32) and Shap-E's conditioner (ViT-L/14) are both OpenAI CLIP models "
                           "trained on the same WIT-400M data: Shap-E outputs may score higher partly because they "
                           "were optimised toward a related embedding space, not only because they look right")
    result["B2"] = b2

    # ---- B3: retention on procedural prompts (existing protocol), auto vs native
    b3r, root = records(prompts, "B3_retention")
    b3 = {"rows": [], "aggregate": {}}
    for r in b3r:
        row = {"asset_id": r.asset_id, "caption": r.caption, "route": routing[r.caption]}
        for cand, gen in ((NATIVE, NATIVE), (AUTO, routed(r.caption))):
            u = U(gen, r.caption)
            if u is None:
                row[cand] = {"status": "omitted"}
                continue
            row[cand] = {**evaluate_record(_Fixed(mesh_or_empty(u), gen), r, root), "generator": gen}
        b3["rows"].append(row)
    for cand in (NATIVE, AUTO):
        rows = [x[cand] for x in b3["rows"] if x[cand].get("status") != "omitted"]
        b3["aggregate"][cand] = {"n": len(rows), "success_rate": mean([float(x["success"]) for x in rows]),
                                 "chamfer_l1_mean": mean([x["chamfer_l1"] for x in rows]),
                                 "fscore@0.05_mean": mean([x["fscore@0.05"] for x in rows]),
                                 "part_count_accuracy": mean([float(x["part_count_correct"]) for x in rows]),
                                 "color_accuracy": (sum(x["color_parts_correct"] for x in rows) /
                                                    max(1, sum(x["color_parts_total"] for x in rows)))}
    result["B3"] = b3

    # ---- B4: composition {native, shap-e} x {single-shot, planner}
    b4r, root = records(prompts, "B4_composition")
    b4 = {"rows": [], "aggregate": {}, "touch_tolerance": 0.02}
    for r in b4r:
        gt = np.load(root / r.files["gt_mesh"])
        gt_touch = touch_components(gt["v"], gt["f"])
        plan, _ = planner.parse(r.caption)
        row = {"asset_id": r.asset_id, "caption": r.caption, "plan": plan.as_dict() if plan else None,
               "gt_components": int(gt["components"]), "gt_touch_components": gt_touch}
        for gen in (NATIVE, SHAP_E):
            for mode in ("single", "planner"):
                name = f"{gen}:{mode}"
                if mode == "single":
                    u = U(gen, r.caption)
                    if u is None:
                        row[name] = {"status": "omitted"}
                        continue
                    out_ = mesh_or_empty(u)
                else:
                    ua, ub = U(gen, plan.a), U(gen, plan.b)
                    if ua is None or ub is None:
                        row[name] = {"status": "omitted"}
                        continue
                    ma, mb = mesh_or_empty(ua), mesh_or_empty(ub)
                    comp = planner.compose(plan, [({"generator": gen}, ma), ({"generator": gen}, mb)])
                    if comp["status"] != "ok":
                        out_ = {"vertices": None, "faces": None, "colors": None,
                                "report": {"valid": False, "seconds": None}}
                    else:
                        out_ = comp["scene"]
                ev = evaluate_record(_Fixed(out_, name), r, root)
                al = align_and_score(out_["vertices"], out_["faces"], gt["v"], gt["f"])
                tc = touch_components(out_["vertices"], out_["faces"])
                row[name] = {**ev, "aligned_chamfer_l1": al.get("chamfer_l1"),
                             "aligned_fscore@0.05": al.get("fscore@0.05"), "touch_components": tc,
                             "touch_part_count_correct": tc == gt_touch if tc is not None else False}
        b4["rows"].append(row)
    for gen in (NATIVE, SHAP_E):
        for mode in ("single", "planner"):
            name = f"{gen}:{mode}"
            rows = [x[name] for x in b4["rows"] if x[name].get("status") != "omitted"]
            b4["aggregate"][name] = {
                "n": len(rows), "success_rate": mean([float(x["success"]) for x in rows]),
                "part_count_accuracy": mean([float(x["part_count_correct"]) for x in rows]),
                "touch_part_count_accuracy": mean([float(x["touch_part_count_correct"]) for x in rows]),
                "chamfer_l1_mean": mean([x["chamfer_l1"] for x in rows]),
                "aligned_chamfer_l1_mean": mean([x["aligned_chamfer_l1"] for x in rows]),
                "color_accuracy": (sum(x["color_parts_correct"] for x in rows) /
                                   max(1, sum(x["color_parts_total"] for x in rows))) if rows else None}
    result["B4"] = b4

    # ---- router accuracy against the intended domain of each benchmark (B1/B2 open-world, B3/B4 procedural)
    intended = {p: (SHAP_E if g in ("B1", "B2") else NATIVE) for g, p in all_benchmark_prompts(prompts)}
    ok = [routing[p]["generator"] == intended[p] for p in intended]
    oracle = [(r["prompt"], SHAP_E if r[SHAP_E]["chamfer_l1"] < r[NATIVE]["chamfer_l1"] else NATIVE)
              for r in both]
    result["routing"] = {"rule": "native iff no content word outside the native vocabulary, else shap-e",
                         "domain_accuracy": f"{sum(ok)}/{len(ok)}",
                         "b1_oracle_agreement": f"{sum(routing[p]['generator'] == g for p, g in oracle)}/{len(oracle)}",
                         "decisions": routing}

    # ---- cost
    sh = [r for r in done.values() if r["gen"] == SHAP_E]
    result["cost"] = {"shap_e_units": len(sh), "shap_e_valid": sum(bool(r.get("valid")) for r in sh),
                      "shap_e_seconds_mean": mean([r.get("seconds") for r in sh]),
                      "shap_e_wall_s_mean": mean([r.get("wall_s") for r in sh]),
                      "shap_e_peak_rss_mib_max": max([r.get("peak_rss_mib") or 0 for r in sh], default=None),
                      "native_seconds_mean": mean([r.get("seconds") for r in done.values() if r["gen"] == NATIVE]),
                      "status_counts": {s: sum(r["status"] == s for r in done.values())
                                        for s in sorted({r["status"] for r in done.values()})}}
    if sheets:
        rows_img = [[*ims] for _, _, ims in sheets]
        save_png(out / "B2_views_gt_native_shap_e.png", contact_sheet(rows_img))
        result["images"] = {"B2_views": str((out / "B2_views_gt_native_shap_e.png").relative_to(ROOT)),
                            "order": [f"{c}: {p}" for c, p, _ in sheets]}
    (ROOT / cfg["result_json"]).write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {cfg['result_json']} ({result['completeness']}; {len(omitted)} omitted units)")
    return result


def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def cmd_report(cfg, prompts, out):
    """E-T3_openworld_cpu.report.md from the result JSON (numbers only from that file)."""
    R = json.loads((ROOT / cfg["result_json"]).read_text())
    L = [f"# E-T3 open-world text-to-3D on CPU — {R['completeness']} run", "",
         f"Config `{CFG.relative_to(ROOT)}`; frozen prompts `{R['prompts_file']}` (sha256 `{R['prompts_sha256'][:16]}…`).",
         (f"Device: CPU ({R['environment'].get('platform')}); torch {R['environment'].get('torch')}; "
         f"git {R['environment'].get('git_commit')}. Training performed: **{R['training_performed']}**."),
         (f"Shap-E: text300M + vector decoder, {cfg['shap_e']['steps']} Karras steps, guidance "
         f"{cfg['shap_e']['guidance']}, seed {cfg['seed']}. Native: BerkeliumTextShape, resolution "
         f"{cfg['native']['resolution']}, adapter {cfg['native']['adapter']}."), ""]
    if R["omitted_units"]:
        L += [f"**Omitted ({len(R['omitted_units'])} generation units, budget):** " +
              ", ".join(f"`{u}`" for u in R["omitted_units"]), ""]
    c = R["cost"]
    L += ["## Cost", "", "| | value |", "|---|---|",
          f"| Shap-E generations run | {c['shap_e_units']} ({c['shap_e_valid']} valid meshes) |",
          f"| Shap-E seconds / generation (in-process) | {_f(c['shap_e_seconds_mean'], 1)} |",
          f"| Shap-E wall seconds / generation (incl. process start) | {_f(c['shap_e_wall_s_mean'], 1)} |",
          f"| Shap-E peak RSS (max over runs) | {c['shap_e_peak_rss_mib_max']} MiB |",
          f"| native seconds / generation | {_f(c['native_seconds_mean'], 2)} |",
          f"| unit status counts | {c['status_counts']} |", ""]
    b1 = R["B1"]
    L += ["## B1 — open-world prompts with real-object ground truth (Google Scanned Objects)", "",
          f"Alignment: {b1['alignment']}. Chamfer-L1 and F-score in unit-extent units.", "",
          "| object | prompt | route | native CD | native F@0.05 | Shap-E CD | Shap-E F@0.05 | Shap-E valid |",
          "|---|---|---|---|---|---|---|---|"]
    for r in b1["rows"]:
        L.append(f"| {r['category']} | {r['prompt']} | {r['route']['generator']} | {_f(r[NATIVE].get('chamfer_l1'))} |"
                 f" {_f(r[NATIVE].get('fscore@0.05'))} | {_f(r[SHAP_E].get('chamfer_l1'))} |"
                 f" {_f(r[SHAP_E].get('fscore@0.05'))} | {r[SHAP_E]['status']} |")
    L += ["", "| generator | n | valid rate | mean CD-L1 | mean F@0.02 | mean F@0.05 |", "|---|---|---|---|---|---|"]
    for k in (NATIVE, SHAP_E, AUTO):
        a = b1["aggregate"][k]
        L.append(f"| {k} | {a['n']} | {_f(a['valid_rate'], 2)} | {_f(a['chamfer_l1_mean'])} | "
                 f"{_f(a['fscore@0.02_mean'])} | {_f(a['fscore@0.05_mean'])} |")
    L += ["", f"Shap-E closer to GT than native (Chamfer): {b1['aggregate']['shap_e_better_chamfer']}.", ""]
    b2 = R["B2"]
    L += ["## B2 — semantic relevance (CLIP R-precision)", "",
          (f"Evaluator: {b2['evaluator']['model']} (sha256 `{b2['evaluator']['sha256'][:16]}…`), 4 views per mesh, "
          f"ranked against all {b2['candidates']} benchmark prompts."), "",
          "| candidate | n | R@1 | R@5 |", "|---|---|---|---|"]
    for r in b2["rows"]:
        L.append(f"| {r['candidate']} | {r['n']} | {_f(r['r_at_1'], 2)} | {_f(r['r_at_5'], 2)} |")
    L += ["", "Per-prompt rank of the true prompt (1 = best):", "", "| prompt | " +
          " | ".join(r["candidate"] for r in b2["rows"]) + " |", "|---|" + "---|" * len(b2["rows"])]
    allp = []
    for r in b2["rows"]:
        for pp in r["per_prompt"]:
            if pp["prompt"] not in allp:
                allp.append(pp["prompt"])
    for pr in allp:
        cells = []
        for r in b2["rows"]:
            m = [x for x in r["per_prompt"] if x["prompt"] == pr]
            cells.append(str(m[0]["rank"]) + ("" if m[0]["status"] in ("ok", "gt") else f" ({m[0]['status']})")
                         if m else "—")
        L.append(f"| {pr} | " + " | ".join(cells) + " |")
    L += ["", f"Residual bias: {b2['residual_bias']}.", ""]
    b3 = R["B3"]
    L += ["## B3 — retention on procedural prompts (existing protocol berkelium-text3d-eval/1)", "",
          "| generator | n | success | mean CD-L1 | mean F@0.05 | part-count acc | colour acc |", "|---|---|---|---|---|---|---|"]
    for k in (NATIVE, AUTO):
        a = b3["aggregate"][k]
        L.append(f"| {k} | {a['n']} | {_f(a['success_rate'], 2)} | {_f(a['chamfer_l1_mean'])} | "
                 f"{_f(a['fscore@0.05_mean'])} | {_f(a['part_count_accuracy'], 2)} | {_f(a['color_accuracy'], 2)} |")
    L += ["", "Routes: " + ", ".join(sorted({r["route"]["generator"] for r in b3["rows"]})) + ".", ""]
    b4 = R["B4"]
    L += ["## B4 — composition: {native, Shap-E} × {single-shot, planner}", "",
          ("CD = Chamfer-L1 in the shared normalized frame (existing protocol, no alignment); aligned CD = B1 "
          "alignment; part count = existing rule (connected components == GT); touch-merged = components after "
          f"merging pieces within {b4['touch_tolerance']} (evaluate.py's 'touching parts merge'), same for GT."), "",
          "| candidate | n | success | part-count acc | touch-merged acc | CD-L1 | aligned CD-L1 | colour acc |",
          "|---|---|---|---|---|---|---|---|"]
    for k, a in b4["aggregate"].items():
        L.append(f"| {k} | {a['n']} | {_f(a['success_rate'], 2)} | {_f(a['part_count_accuracy'], 2)} | "
                 f"{_f(a['touch_part_count_accuracy'], 2)} | {_f(a['chamfer_l1_mean'])} | "
                 f"{_f(a['aligned_chamfer_l1_mean'])} | {_f(a['color_accuracy'], 2)} |")
    L += ["", "| asset | caption | GT comps | " + " | ".join(b4["aggregate"]) + " |",
          "|---|---|---|" + "---|" * len(b4["aggregate"])]
    for r in b4["rows"]:
        cells = []
        for k in b4["aggregate"]:
            x = r[k]
            cells.append("omitted" if x.get("status") == "omitted" else
                         f"{x.get('components', '—')}c / CD {_f(x.get('chamfer_l1'))}")
        L.append(f"| {r['asset_id']} | {r['caption']} | {r['gt_components']} | " + " | ".join(cells) + " |")
    ro = R["routing"]
    L += ["", "## Router", "", (f"Rule: {ro['rule']}. Domain accuracy (B1/B2 → Shap-E, B3/B4 → native): "
          f"{ro['domain_accuracy']}. Agreement with the better generator on B1 (Chamfer oracle): "
          f"{ro['b1_oracle_agreement']}."), ""]
    (ROOT / cfg["result_json"]).with_suffix("").with_suffix(".report.md").write_text("\n".join(L) + "\n")
    print("wrote", (ROOT / cfg["result_json"]).with_suffix("").with_suffix(".report.md").relative_to(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["plan", "generate", "score", "report"])
    a = ap.parse_args(argv)
    cfg, prompts = load_cfg()
    out = ROOT / cfg["out"]
    {"plan": cmd_plan, "generate": cmd_generate, "score": cmd_score, "report": cmd_report}[a.phase](cfg, prompts,
                                                                                                    out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
