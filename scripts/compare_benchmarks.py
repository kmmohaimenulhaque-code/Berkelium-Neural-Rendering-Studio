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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.renderer.classify import (
    CLASSIFICATION_VERSION,
    classify_case,
    gate_summary,
    identical_oracle_errors,
    oracle_role,
)

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


def _correctness_status(c: dict | None, name: str) -> str:
    if (c or {}).get("status") != "measured":
        return "not_run"
    if name == "gradient_bitwise_repeatable":
        return "pass" if c.get("value") == 1.0 else "fail"
    if c.get("oracle_role") == "diagnostic":
        tol = c.get("tolerance")
        within = c.get("within_tolerance")
        side = "within" if within else ("beyond" if within is False else "vs")
        extra = (f"; {c.get('violations')}/{c.get('entries')} entries beyond spec tol" if "violations" in c else "")
        return f"diagnostic: FP64 oracle not a gate for this case ({side} tol {tol}{extra})"
    cp = c.get("passed")
    if cp is None:     # measured but not a gate (e.g. large-scene agreement)
        return (f"info: not a gate ({c.get('gate', 'no tolerance defined')}; "
                f"{c.get('violations', 0)}/{c.get('entries', '?')} entries beyond spec tol)")
    return "pass" if cp else "fail"


def _cross_backend_rows(case: dict, bb: str, cb: str) -> list[dict]:
    cross = case.get("cross_backend")
    if isinstance(cross, dict) and cb in cross:
        rows = []
        for x in cross[cb]["measurements"]:
            r = _row(x["name"], "correctness", None, x)
            r["status"] = ("info: bitwise equal" if x.get("bitwise_equal") else
                           f"info: {x.get('n_differing')}/{x.get('entries')} entries differ")
            r["reason"] = "direct comparison, no oracle"
            rows.append({"case": case["case"], **r})
        return rows
    reason = (cross or {}).get("reason") if isinstance(cross, dict) else None
    reason = reason or ("not recorded: this run predates the direct cross-backend output comparison; it needs a "
                        "rerun on the GPU and is not inferred from oracle errors")
    return [{"case": case["case"], "metric": f"forward_output_diff_{cb}_vs_{bb}", "group": "correctness",
             "baseline": None, "candidate": None, "abs_diff": None, "rel_change": None, "baseline_status": "not_run",
             "candidate_status": "not_run", "status": "not_run", "reason": reason}]


def authorization(res: dict) -> dict:
    """Static approval from the experiment config, and the per-run CLI authorization only if the runner recorded it.
    Never inferred into a recorded approval."""
    exp = res.get("experiment", {})
    rec = res.get("authorization")
    if rec:
        return rec
    return {"static_config_approved": bool(exp.get("approved")), "cli_i_approve_gpu_cost": "not_recorded",
            "authorized_by": "not_recorded",
            "note": ("this result predates runtime authorization metadata. The runner refuses an experiment whose "
                     "config has approved=false unless --i-approve-gpu-cost is passed, so a completed run implies the "
                     "flag was used — an inference from the runner code, not a recorded fact (the source tree "
                     f"{res.get('environment', {}).get('git_commit')} may differ from this commit)")
            if not exp.get("approved") and res.get("status") == "completed" else ""}


def compare_renderer(res):
    proto = res["config"]
    bb, cb = proto["baseline_backend"], proto["candidate_backend"]
    cases = [classify_case(c) for c in res["cases"]]          # recorded values untouched; roles applied
    out, case_notes = [], []
    for case in cases:
        B = {x["name"]: x for x in case["backends"].get(bb, {}).get("measurements", [])}
        C = {x["name"]: x for x in case["backends"].get(cb, {}).get("measurements", [])}
        for name in sorted(set(B) | set(C), key=lambda n: ((B.get(n) or C.get(n))["group"], n)):
            b, c = B.get(name), C.get(name)
            grp = (b or c)["group"]
            gate = _correctness_status(c, name) if grp == "correctness" else None
            out.append({"case": case["case"], **_row(name, grp, b, c, gate=gate)})
        out += _cross_backend_rows(case, bb, cb)
        case_notes.append({"case": case["case"], "oracle_role": oracle_role(case["case"]),
                           "forward_oracle_errors_bitwise_identical": identical_oracle_errors(case, bb, cb)})
    q = res.get("quality", {})
    if q:
        out.append({"case": "fit", **_row("heldout_view_psnr_after_fit", "quality", q.get(bb), q.get(cb), hib=True)})
    return {"baseline": bb, "candidate": cb, "rows": out, "classification": CLASSIFICATION_VERSION,
            "case_notes": case_notes, "gate_summary": gate_summary(cases), "authorization": authorization(res),
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
