"""Role of the FP64 oracle per benchmark case, applied identically to new runs and to recorded results.

Protocol (benchmarks/configs/renderer_*.json): "FP64 CPU reference is the oracle for correctness on the small cases
only". A case marked ``"gate": true`` (the spec's smooth fixture) is an ACCEPTANCE GATE: every oracle metric must be
within the spec tolerance. On every other case the oracle is DIAGNOSTIC: the raw error, the tolerance and whether
the error is within it are kept, but ``passed`` is ``None`` and the row is never reported as an acceptance failure.

Why the oracle cannot gate large cases (evidence: results/renderer/E-R1_forward_diagnosis_cpu.json): profile
gsplat_v1 has hard decision boundaries (skip when alpha < 1/255, stop when T*(1-alpha) <= 1e-4). Any FP32
implementation evaluates a few Gaussians on the other side of a boundary than FP64 does, and one flip changes a
pixel by up to ~(1/255)*T*|colour difference| (about 4e-3), independent of how accurate the arithmetic is. Such
near-boundary events become likely as the workload grows, so a max-abs criterion against the FP64 oracle measures
how many Gaussians sit near a threshold, not whether a backend is correct.

This module never changes a recorded value. ``classify_case`` returns new dicts; input records are not mutated.
"""

from __future__ import annotations

import copy

FORWARD_ORACLE_METRICS = ("forward_rgb_max_abs_err", "forward_alpha_max_abs_err", "forward_depth_max_abs_err_alpha>0.5")
CLASSIFICATION_VERSION = "oracle-role-v1"
DIAGNOSTIC_NOTE = ("FP64 oracle is diagnostic (not an acceptance gate) for this case per protocol: FP32 vs FP64 "
                   "decisions at the 1/255 alpha and 1e-4 transmittance thresholds can flip near-boundary Gaussians; "
                   "raw value and tolerance preserved")


def oracle_role(case: dict) -> str:
    return "acceptance_gate" if case.get("gate") else "diagnostic"


def is_oracle_metric(name: str) -> bool:
    return name in FORWARD_ORACLE_METRICS or (name.startswith("grad_") and name.endswith("_max_abs_err"))


def _within_tolerance(meas: dict) -> bool | None:
    """Raw outcome against the recorded tolerance, from recorded fields only."""
    if "within_tolerance" in meas:
        return meas["within_tolerance"]
    if "violations" in meas:
        return meas["violations"] == 0
    if isinstance(meas.get("passed"), bool):          # forward rows recorded passed = (err <= tol) in every case
        return meas["passed"]
    tol, v = meas.get("tolerance"), meas.get("value")
    if isinstance(tol, (int, float)) and isinstance(v, (int, float)):
        return v <= tol
    return None


def classify_measurement(meas: dict, case: dict) -> dict:
    out = copy.deepcopy(meas)
    if not is_oracle_metric(meas.get("name", "")) or meas.get("status") != "measured":
        return out
    role = oracle_role(case)
    within = _within_tolerance(meas)
    out["oracle_role"] = role
    out["within_tolerance"] = within
    out["classification"] = CLASSIFICATION_VERSION
    if role == "acceptance_gate":
        out["passed"] = within
        out.setdefault("gate", "spec_smooth_fixture")
    else:
        if meas.get("passed") is not None:
            out["recorded_passed"] = meas["passed"]   # the original (mis)classification stays visible
        out["passed"] = None
        out["gate"] = "diagnostic_oracle" if meas["name"] in FORWARD_ORACLE_METRICS else meas.get("gate",
                                                                                                 "agreement_only")
        out["classification_note"] = DIAGNOSTIC_NOTE
    return out


def classify_case(case_result: dict) -> dict:
    """-> deep copy of one recorded case with every backend's oracle metrics classified."""
    out = copy.deepcopy(case_result)
    for b in out.get("backends", {}).values():
        b["measurements"] = [classify_measurement(x, out["case"]) for x in b.get("measurements", [])]
    return out


def gate_summary(cases: list[dict]) -> dict:
    """Acceptance gates vs diagnostics over classified cases (counts only; values stay in the rows)."""
    s = {"acceptance_gates": {"pass": 0, "fail": 0, "rows": []}, "diagnostic_beyond_tolerance": []}
    for c in cases:
        for bname, b in c.get("backends", {}).items():
            for x in b.get("measurements", []):
                if "oracle_role" not in x:
                    continue
                if x["oracle_role"] == "acceptance_gate":
                    s["acceptance_gates"]["pass" if x["passed"] else "fail"] += 1
                    if not x["passed"]:
                        s["acceptance_gates"]["rows"].append({"case": c["case"], "backend": bname, "metric": x["name"]})
                elif x.get("within_tolerance") is False:
                    s["diagnostic_beyond_tolerance"].append({"case": c["case"], "backend": bname, "metric": x["name"],
                                                             "value": x["value"], "tolerance": x.get("tolerance")})
    return s


def identical_oracle_errors(case_result: dict, baseline: str, candidate: str) -> dict:
    """Indirect, recorded-data-only evidence about the forward pass: whether the two backends' forward errors vs the
    same oracle are bitwise-identical floats. Equal errors are consistent with identical forward outputs but do not
    prove them (a direct output comparison is required for that)."""
    B = {x["name"]: x for x in case_result["backends"].get(baseline, {}).get("measurements", [])}
    C = {x["name"]: x for x in case_result["backends"].get(candidate, {}).get("measurements", [])}
    out = {}
    for n in FORWARD_ORACLE_METRICS:
        if n in B and n in C and B[n].get("status") == C[n].get("status") == "measured":
            out[n] = B[n]["value"] == C[n]["value"]
    return out
