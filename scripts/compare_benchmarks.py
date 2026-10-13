#!/usr/bin/env python3
"""Compare matched baseline vs candidate results -> results/<bench>/<id>.comparison.json

  python scripts/compare_benchmarks.py results/renderer/E-R0_cpu.json
  python scripts/compare_benchmarks.py results/text3d/E-T1_cpu.json

Renderer: per case, baseline_backend vs candidate_backend from the protocol (same scene/camera/resolution).
Text-to-3D: per comparison id, baseline vs candidate on the same split and prompts.
Rows: metric, group, baseline, candidate, abs_diff, rel_change (only when the baseline is non-zero and the metric
is ratio-scaled), status. Correctness rows carry pass/fail; quality rows improved/regressed/equal; missing values
stay null with their not_run/failed reason.
"""
import json
import sys
from pathlib import Path

RATIO_UNITS = {"ms", "MiB", "s/prompt", "dB", "normalized_object", "normalized_object^2", ""}


def _row(name, group, b, c, hib=None, gate=None):
    bv, cv = (b or {}).get("value"), (c or {}).get("value")
    row = {"metric": name, "group": group, "baseline": bv, "candidate": cv, "abs_diff": None, "rel_change": None,
           "baseline_status": (b or {}).get("status", "missing"), "candidate_status": (c or {}).get("status", "missing"),
           "reason": "; ".join(dict.fromkeys(x for x in ((b or {}).get("reason"), (c or {}).get("reason")) if x))}
    if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
        row["abs_diff"] = cv - bv
        if bv != 0 and (b or {}).get("unit", "") in RATIO_UNITS and name not in ("gradient_bitwise_repeatable",):
            row["rel_change"] = (cv - bv) / abs(bv)
    if gate is not None:
        row["status"] = gate
    elif hib is not None and row["abs_diff"] is not None:
        row["status"] = "equal" if row["abs_diff"] == 0 else ("improved" if (row["abs_diff"] > 0) == hib else "regressed")
    else:
        row["status"] = "info" if row["abs_diff"] is not None else "not_comparable"
    return row


def compare_renderer(res):
    proto = res["config"]
    bb, cb = proto["baseline_backend"], proto["candidate_backend"]
    out = []
    for case in res["cases"]:
        B = {x["name"]: x for x in case["backends"].get(bb, {}).get("measurements", [])}
        C = {x["name"]: x for x in case["backends"].get(cb, {}).get("measurements", [])}
        for name in sorted(set(B) | set(C), key=lambda n: ((B.get(n) or C.get(n))["group"], n)):
            b, c = B.get(name), C.get(name)
            grp = (b or c)["group"]
            gate = None
            if grp == "correctness":
                cp = (c or {}).get("passed")
                if name == "gradient_bitwise_repeatable":
                    cp = (c or {}).get("value") == 1.0
                if (c or {}).get("status") != "measured":
                    gate = "not_run"
                elif cp is None:   # measured but not a gate in this case (e.g. large-scene agreement)
                    gate = f"info: not a gate ({(c or {}).get('gate', 'no tolerance defined')}; " \
                           f"{(c or {}).get('violations', 0)}/{(c or {}).get('entries', '?')} entries beyond spec tol)"
                else:
                    gate = "pass" if cp else "fail"
            out.append({"case": case["case"], **_row(name, grp, b, c, gate=gate)})
    q = res.get("quality", {})
    if q:
        out.append({"case": "fit", **_row("heldout_view_psnr_after_fit", "quality", q.get(bb), q.get(cb), hib=True)})
    return {"baseline": bb, "candidate": cb, "rows": out,
            "global_correctness": res.get("correctness_global", [])}


def paired_bootstrap(base_recs, cand_recs, key, n_boot=2000, seed=0):
    """Per-asset paired difference (candidate - baseline) with a seeded 95% bootstrap CI."""
    import random
    b = {r["asset_id"]: r.get(key) for r in base_recs}
    d = [r.get(key) - b[r["asset_id"]] for r in cand_recs
         if isinstance(r.get(key), (int, float)) and isinstance(b.get(r["asset_id"]), (int, float))]
    if len(d) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {"n_pairs": len(d), "mean_delta": sum(d) / len(d), "ci95": [means[int(0.025 * n_boot)],
            means[int(0.975 * n_boot) - 1]], "method": f"paired bootstrap, {n_boot} resamples, seed {seed}"}


def compare_text3d(res):
    comps = []
    for c in res["comparisons"]:
        B = {x["name"]: x for x in c["baseline"]["measurements"]}
        C = {x["name"]: x for x in c["candidate"]["measurements"]}
        rows = [_row(n, (B.get(n) or C.get(n))["group"], B.get(n), C.get(n),
                     hib=(B.get(n) or C.get(n)).get("higher_is_better")) for n in B]
        for key in ("chamfer_l1", "fscore@0.05", "view_psnr", "normal_consistency"):
            pb = paired_bootstrap(c["baseline"].get("records", []), c["candidate"].get("records", []), key)
            if pb:
                rows.append({"metric": f"paired_delta_{key}", "group": "quality", "baseline": None,
                             "candidate": pb["mean_delta"], "abs_diff": pb["mean_delta"], "rel_change": None,
                             "status": "CI excludes 0" if pb["ci95"][0] > 0 or pb["ci95"][1] < 0 else "CI includes 0",
                             "reason": f"95% CI [{pb['ci95'][0]:.4g}, {pb['ci95'][1]:.4g}], n={pb['n_pairs']}",
                             "baseline_status": "measured", "candidate_status": "measured"})
        comps.append({"id": c["id"], "split": c["split"], "baseline": c["baseline"].get("generator", c["baseline"]
                      .get("status")), "candidate": c["candidate"].get("generator", c["candidate"].get("status")),
                      "rows": rows})
    return {"comparisons": comps}


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        print(__doc__)
        return 2
    p = Path(argv[0])
    res = json.loads(p.read_text())
    if res.get("status") != "completed":
        print(f"cannot compare: run status is {res.get('status')} ({res.get('error')})")
        return 1
    cmp = compare_renderer(res) if res["benchmark"] == "renderer" else compare_text3d(res)
    cmp.update({"source": str(p), "config_hash": res["config_hash"], "environment": res["environment"]})
    out = p.with_suffix(".comparison.json")
    out.write_text(json.dumps(cmp, indent=2, default=str))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
