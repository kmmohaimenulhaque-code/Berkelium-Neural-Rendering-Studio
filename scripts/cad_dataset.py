#!/usr/bin/env python3
"""Build the verified E-C1 compiler dataset ``berkelium-cad-v1`` (runs on Kaggle; needs the Berkelium core).

  python scripts/cad_dataset.py build --out data/cad_v1 --per-family 520 --workers 4
  python scripts/cad_dataset.py smoke                     # 2 samples per family, quick sanity check

Rows: procedural part families (services/neural/berkelium_neural/cad/families.py), the core's spur-gear pair CEM
(its own sampler and intent renderer), a cantilever-beam CEM slice and the core's lantern design. Every target goes
through ``berkelium.pipeline.run`` (Manifold backend) and is kept only if the record validates (summary pass/warn),
geometry is realised (L3 pass) and every requirement passes. Rejections are kept with their reason, never trained
on. Splits: seen families -> train/val/test by item hash; HELDOUT_FAMILIES -> ``heldout`` only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "neural"))

from berkelium_neural.cad import CORE_COMMIT  # noqa: E402
from berkelium_neural.cad import families as FAM  # noqa: E402
from berkelium_neural.cad.prompts import PROMPT_VERSION  # noqa: E402

GEAR = "gear.spur_pair@0.1"
BEAM = "structure.cantilever_beam@0.1"


def split_of(key: str, heldout: bool) -> str:
    if heldout:
        return "heldout"
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 92 else "val" if h < 95 else "test"


# ----------------------------------------------------------------------------------------- candidate generators
def gen_family(args):
    fam, seed, vague = args
    r = FAM.sample(fam, seed, vague)
    if r is None:
        return {"family": fam, "seed": seed, "rejected": "generator constraint"}
    text, prop = r
    return {"family": fam, "seed": seed, "vague": vague, "text": text, "target": prop}


def gen_gear(seed: int, mode: str):
    import numpy as np
    from berkelium.cem.protocol import default_registry
    from berkelium.datasets.factory import plan_target, requirements_from_params
    from berkelium.datasets.intents import gear_intent
    r = np.random.default_rng(seed)
    p = default_registry().get(GEAR).sample(r, mode)
    req = requirements_from_params(p, r)
    return {"family": "gear_pair", "seed": seed, "vague": False, "text": gear_intent(req, r),
            "target": plan_target(req)}


MATERIALS = [("steel", 200), ("structural steel", 200), ("aluminium", 69), ("aluminum", 69), ("titanium", 114),
             ("stainless steel", 193)]


def gen_beam(seed: int):
    import numpy as np
    r = np.random.default_rng(seed)
    mat, E = MATERIALS[int(r.integers(len(MATERIALS)))]
    L = float(round(r.uniform(100, 2000), -1))
    P = float(round(r.uniform(50, 5000), -1))
    defl = float(round(L / float(r.choice([100, 200, 250, 360, 500])), 2))
    sig = float(round(r.uniform(60, 250) * (E / 200) ** 0.5, 0))
    ar = float(r.choice([1.0, 1.5, 2.0, 3.0]))
    req = {"tip_load_N": P, "length_mm": L, "max_tip_deflection_mm": defl, "allowable_stress_MPa": sig,
           "youngs_modulus_GPa": float(E), "aspect_ratio": ar}
    opener = str(r.choice(["Size a cantilever beam", "Design a rectangular cantilever", "I need a cantilever bracket arm",
                           "Make a cantilever beam", "Size a rectangular bar fixed at one end"]))
    text = (f"{opener} in {mat} (E = {E} GPa), {L:g} mm long, carrying {P:g} N at the tip, with tip deflection at "
            f"most {defl:g} mm, stress below {sig:g} MPa and a height/width ratio of {ar:g}.")
    tgt = {"schema_version": "0.1.0", "kind": "design_proposal",
           "intent": {"summary": "rectangular cantilever beam", "assumptions": [], "open_questions": []},
           "specification": {"requirements": [], "parameters": []},
           "structure": {"components": [{"id": "beam", "kind": "cem", "cem": BEAM, "requirements": req}]}}
    return {"family": "cantilever_beam", "seed": seed, "vague": False, "text": text, "target": tgt}


def gen_lantern(seed: int):
    import numpy as np
    from berkelium.designs.procedural import lantern_proposal
    r = np.random.default_rng(seed)
    h, rad = float(np.round(r.uniform(80, 200), 0)), float(np.round(r.uniform(30, 70), 0))
    t, slots = float(np.round(r.uniform(2, 5), 1)), int(r.integers(8, 25))
    prop = lantern_proposal(h, rad, t, slots).model_dump(mode="json", exclude_none=True)
    prop["intent"].pop("prompt", None)
    text = str(r.choice([
        f"Model a lantern-like vessel about {h:g} mm tall and {2 * rad:g} mm across with a {t:g} mm wall, a bulging "
        f"profile, a ring of {slots} tilted slots and a ring handle on top.",
        f"A decorative lantern, {h:g} mm high, {2 * rad:g} mm wide, {t:g} mm wall, {slots} angled slots around the "
        f"belly and a handle ring.",
        f"Make a {h:g} mm tall perforated lantern with a {rad:g} mm radius, {slots} slanted slots and a top ring, "
        f"walls {t:g} mm."]))
    return {"family": "lantern", "seed": seed, "vague": False, "text": text, "target": prop}


# ------------------------------------------------------------------------------------------------ verification
def verify(row: dict) -> dict:
    if "rejected" in row:
        return row
    from berkelium.pipeline import run
    t0 = time.time()
    try:
        res = run(row["target"], realize=True, backend="manifold")
    except Exception as e:                                  # a crash in the core is a rejection, never a pass
        return {**row, "rejected": f"core exception {type(e).__name__}: {str(e)[:200]}"}
    rec = res.record
    if rec is None:
        return {**row, "rejected": "schema: " + json.dumps(res.schema_errors[:2], default=str)[:300]}
    v = rec.evaluation.validation
    reqs = [r for r in v.results if r.validator.startswith("requirement.")]
    l3 = [r for r in v.results if r.level == 3]
    if v.summary not in ("pass", "warn"):
        bad = [f"L{r.level} {r.target}: {r.message}" for r in v.results if r.status in ("fail", "error")][:3]
        return {**row, "rejected": f"summary {v.summary}: {bad}"}
    if any(r.status != "pass" for r in reqs):
        return {**row, "rejected": "requirement not met: " + "; ".join(r.message for r in reqs if r.status != "pass")}
    if not any(r.status == "pass" for r in l3):
        return {**row, "rejected": "geometry not realised"}
    vol = [m.value.value for m in rec.evaluation.measurements if m.name == "volume"]
    bbox = [m.vector for m in rec.evaluation.measurements if m.name == "bbox_min_max_mm"]
    row["verification"] = {"summary": v.summary, "record_hash": rec.content_hash, "requirements": len(reqs),
                           "volume_mm3": vol[0] if vol else None, "bbox": bbox[0] if bbox else None,
                           "seconds": round(time.time() - t0, 3)}
    return row


def candidates(per_family: int, custom_extra: int, gears: int, beams: int, lanterns: int, seed: int):
    out = []
    for i, fam in enumerate(FAM.FAMILIES):
        n = per_family + (custom_extra if fam == "custom_part" else 0)
        if fam in FAM.HELDOUT_FAMILIES:
            n = 90
        for k in range(n):
            s = seed * 1_000_003 + i * 100_000 + k
            out.append(("fam", (fam, s, (k % 7) == 3)))
    modes = ["valid"] * 4 + ["boundary"]
    out += [("gear", (seed * 7919 + k, modes[k % 5])) for k in range(gears)]
    out += [("beam", (seed * 104_729 + k,)) for k in range(beams)]
    out += [("lantern", (seed * 15_485_863 + k,)) for k in range(lanterns)]
    return out


def make(c):
    kind, a = c
    try:
        row = {"fam": gen_family, "gear": lambda x: gen_gear(*x), "beam": lambda x: gen_beam(*x),
               "lantern": lambda x: gen_lantern(*x)}[kind](a)
    except Exception as e:
        return {"family": kind, "seed": a[0] if kind != "fam" else a[1], "rejected": f"generator {e!r}"[:300]}
    return verify(row)


def cmd_build(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = candidates(a.per_family, a.custom_extra, a.gears, a.beams, a.lanterns, a.seed)
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(make, cands, chunksize=8)):
            rows.append(r)
            if (i + 1) % 1000 == 0:
                ok = sum("rejected" not in x for x in rows)
                print(f"{i + 1}/{len(cands)} done, {ok} verified, {time.time() - t0:.0f}s", flush=True)
    acc = [r for r in rows if "rejected" not in r]
    rej = [r for r in rows if "rejected" in r]
    seen = set()
    by_split: dict[str, list] = {}
    for r in sorted(acc, key=lambda r: (r["family"], r["seed"], r.get("vague", False))):
        key = r["text"]
        if key in seen:
            continue
        seen.add(key)
        r["id"] = hashlib.sha256(f"{r['family']}/{r['seed']}/{r.get('vague')}".encode()).hexdigest()[:20]
        r["split"] = split_of(r["id"], r["family"] in FAM.HELDOUT_FAMILIES)
        r["prompt_version"] = PROMPT_VERSION
        by_split.setdefault(r["split"], []).append(r)
    files = {}
    for split, rs in sorted(by_split.items()):
        data = "".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in rs).encode()
        (out / f"{split}.jsonl.gz").write_bytes(gzip.compress(data, mtime=0))
        files[f"{split}.jsonl.gz"] = {"rows": len(rs), "sha256_uncompressed": hashlib.sha256(data).hexdigest()}
    (out / "rejected.jsonl.gz").write_bytes(gzip.compress(
        "".join(json.dumps(x, sort_keys=True, default=str) + "\n" for x in rej).encode(), mtime=0))
    fam_counts = {s: dict(Counter(r["family"] for r in rs)) for s, rs in by_split.items()}
    rej_fam = Counter(r["family"] for r in rej)
    reasons = Counter((r["family"], str(r["rejected"])[:90]) for r in rej)
    manifest = {"dataset": "berkelium-cad-v1", "prompt_version": PROMPT_VERSION, "core_commit": CORE_COMMIT,
                "config": vars(a) | {"func": None}, "candidates": len(cands), "verified": len(acc),
                "unique": sum(len(v) for v in by_split.values()), "rejected": len(rej), "files": files,
                "heldout_families": list(FAM.HELDOUT_FAMILIES), "family_counts": fam_counts,
                "rejected_by_family": dict(rej_fam),
                "top_rejections": [{"family": f, "reason": m, "n": n} for (f, m), n in reasons.most_common(40)],
                "vague_fraction": round(sum(r.get("vague", False) for r in acc) / max(1, len(acc)), 3),
                "seconds": round(time.time() - t0, 1)}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({k: manifest[k] for k in ("candidates", "verified", "unique", "rejected", "files")}, indent=1))
    return 0


def cmd_smoke(a) -> int:
    fails = 0
    for fam in FAM.FAMILIES:
        for k, vague in ((1, False), (2, True)):
            r = make(("fam", (fam, 1000 + k, vague)))
            ok = "rejected" not in r
            fails += not ok
            print(f"{fam:16s} vague={vague!s:5s} {'OK ' if ok else 'REJ'} "
                  f"{r.get('verification', {}).get('seconds', '')} {'' if ok else r['rejected'][:160]}")
            if ok and a.show:
                print("   ", r["text"])
    for kind, args in (("gear", (1, "valid")), ("beam", (3,)), ("lantern", (5,))):
        r = make((kind, args))
        print(f"{kind:16s} {'OK ' if 'rejected' not in r else 'REJ ' + r['rejected'][:200]}")
        if a.show and "text" in r:
            print("   ", r["text"])
    return 0 if fails == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="data/cad_v1")
    b.add_argument("--per-family", type=int, default=520)
    b.add_argument("--custom-extra", type=int, default=2500)
    b.add_argument("--gears", type=int, default=700)
    b.add_argument("--beams", type=int, default=400)
    b.add_argument("--lanterns", type=int, default=150)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    s = sub.add_parser("smoke")
    s.add_argument("--show", action="store_true")
    a = ap.parse_args()
    return {"build": cmd_build, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
