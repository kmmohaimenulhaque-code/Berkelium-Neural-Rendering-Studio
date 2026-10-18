"""Benchmark classification, comparison, reporting and run-authorization logic (CPU only, no GPU, no new runs).

The FP64 oracle is an acceptance gate only on protocol cases marked ``gate``; elsewhere it is diagnostic. These tests
pin that the strict small-case gate is unchanged, that diagnostic rows keep their raw values, that recorded results
are never modified, and that per-run CLI authorization is recorded without rewriting the static config."""

import copy
import hashlib
import json
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.renderer.classify import (
    classify_case,
    classify_measurement,
    gate_summary,
    identical_oracle_errors,
    oracle_role,
)

GATE = {"n": 16, "res": 32, "seed": 5, "gate": True}
LARGE = {"n": 100000, "res": 512, "seed": 3}
E_R1 = ROOT / "results" / "renderer" / "E-R1_mi300x.json"


def fwd(value, passed, name="forward_rgb_max_abs_err"):
    return {"name": name, "value": value, "unit": "", "status": "measured", "reason": "", "group": "correctness",
            "tolerance": 2e-4, "passed": passed}


def grad(value, violations, gate):
    return {"name": "grad_means_max_abs_err", "value": value, "unit": "", "status": "measured", "reason": "",
            "group": "correctness", "tolerance": "2e-05+0.002|ref|", "violations": violations, "entries": 48,
            "passed": (violations == 0) if gate else None, "gate": "spec_smooth_fixture" if gate else "agreement_only"}


def legacy_result(gate_rgb=4.5e-7, large_rgb=1.0456e-3, cross=None):
    """Shaped like the recorded E-R1 output (legacy: forward rows carry passed on every case)."""
    def backend(rgb_gate, rgb_large, case):
        big = case is LARGE
        v = rgb_large if big else rgb_gate
        return {"status": "measured", "measurements": [
            fwd(v, v <= 2e-4), grad(0.2 if big else 2e-5, 2868 if big else 0, not big),
            {"name": "gradient_bitwise_repeatable", "value": 1.0, "unit": "", "status": "measured", "reason": "",
             "group": "correctness"},
            {"name": "forward_latency", "value": 1.0, "unit": "ms", "status": "measured", "reason": "",
             "group": "performance"}]}
    cases = []
    for c in (GATE, LARGE):
        case = {"case": c, "scene_sha256": "x", "backends": {"gsplat": backend(gate_rgb, large_rgb, c),
                                                             "berkelium_rocm": backend(gate_rgb, large_rgb, c)}}
        if cross is not None:
            case["cross_backend"] = cross
        cases.append(case)
    return {"experiment": {"id": "E-T", "title": "t", "approved": False}, "status": "completed",
            "benchmark": "renderer", "semantics": "s", "config_hash": "0" * 64, "method": "m", "started_at": "t",
            "environment": {"git_commit": "abc+dirty"}, "correctness_global": [],
            "config": {"baseline_backend": "gsplat", "candidate_backend": "berkelium_rocm", "cases": [GATE, LARGE]},
            "cases": cases, "quality": {}}


# ------------------------------------------------------------------------------------------------ classification
def test_oracle_role_follows_protocol_gate_flag():
    assert oracle_role(GATE) == "acceptance_gate"
    assert oracle_role(LARGE) == "diagnostic"
    assert oracle_role({"n": 1000, "res": 128, "seed": 1, "gate": False}) == "diagnostic"


def test_small_case_gate_stays_strict():
    bad = classify_measurement(fwd(3e-4, False), GATE)
    assert bad["passed"] is False and bad["oracle_role"] == "acceptance_gate" and bad["value"] == 3e-4
    ok = classify_measurement(fwd(1e-7, True), GATE)
    assert ok["passed"] is True
    g = classify_measurement(grad(1e-3, 3, True), GATE)
    assert g["passed"] is False and g["within_tolerance"] is False


def test_large_case_is_diagnostic_and_preserves_raw_values():
    raw = fwd(1.0456162494575927e-3, False)
    before = copy.deepcopy(raw)
    out = classify_measurement(raw, LARGE)
    assert raw == before, "input record must not be mutated"
    assert out["value"] == 1.0456162494575927e-3 and out["tolerance"] == 2e-4      # not loosened
    assert out["passed"] is None and out["within_tolerance"] is False and out["recorded_passed"] is False
    assert out["oracle_role"] == "diagnostic" and out["gate"] == "diagnostic_oracle"
    assert "1/255" in out["classification_note"]


def test_non_oracle_metrics_untouched():
    perf = {"name": "forward_latency", "value": 1.2, "unit": "ms", "status": "measured", "reason": "",
            "group": "performance"}
    assert classify_measurement(perf, LARGE) == perf
    not_run = {**fwd(None, None), "status": "not_run", "reason": "DEVICE_UNAVAILABLE"}
    assert classify_measurement(not_run, LARGE) == not_run


def test_gate_summary_and_identical_errors():
    res = legacy_result()
    cases = [classify_case(c) for c in res["cases"]]
    s = gate_summary(cases)
    assert s["acceptance_gates"] == {"pass": 4, "fail": 0, "rows": []}
    assert {d["metric"] for d in s["diagnostic_beyond_tolerance"]} == {"forward_rgb_max_abs_err",
                                                                        "grad_means_max_abs_err"}
    assert identical_oracle_errors(res["cases"][1], "gsplat", "berkelium_rocm") == {"forward_rgb_max_abs_err": True}


# ------------------------------------------------------------------------------------------------ compare + report
def _compare(res):
    from compare_benchmarks import compare_renderer
    return compare_renderer(res)


def test_compare_reports_diagnostic_not_failure_and_keeps_gate_failures():
    res = legacy_result(gate_rgb=3e-4)                       # make the strict gate fail on purpose
    src = json.dumps(res, sort_keys=True)
    cmp = _compare(res)
    assert json.dumps(res, sort_keys=True) == src, "compare must not modify the recorded result"
    rows = {(r["case"]["n"], r["metric"]): r for r in cmp["rows"] if isinstance(r["case"], dict)}
    assert rows[(16, "forward_rgb_max_abs_err")]["status"] == "fail"
    big = rows[(100000, "forward_rgb_max_abs_err")]
    assert big["status"].startswith("diagnostic") and "beyond tol" in big["status"]
    assert big["candidate"] == 1.0456e-3
    assert cmp["gate_summary"]["acceptance_gates"]["fail"] == 2


def test_compare_legacy_cross_backend_is_not_run_not_inferred():
    cmp = _compare(legacy_result())
    xs = [r for r in cmp["rows"] if r["metric"].startswith("forward_output_diff")]
    assert len(xs) == 2 and all(r["status"] == "not_run" and r["candidate"] is None for r in xs)
    assert "rerun" in xs[0]["reason"]


def test_compare_uses_recorded_cross_backend_rows():
    cross = {"berkelium_rocm": {"baseline": "gsplat", "measurements": [
        {"name": "forward_rgb_max_abs_diff_vs_gsplat", "value": 0.0, "unit": "", "status": "measured", "reason": "",
         "group": "correctness", "bitwise_equal": True, "n_differing": 0, "entries": 12, "passed": None}]}}
    rows = [r for r in _compare(legacy_result(cross=cross))["rows"] if r["metric"].endswith("_vs_gsplat")]
    assert len(rows) == 2 and all(r["status"] == "info: bitwise equal" and r["candidate"] == 0.0 for r in rows)


def test_compare_authorization_is_not_retroactively_claimed():
    a = _compare(legacy_result())["authorization"]
    assert a["static_config_approved"] is False
    assert a["cli_i_approve_gpu_cost"] == "not_recorded" and a["authorized_by"] == "not_recorded"
    assert "inference" in a["note"]
    rec = legacy_result()
    rec["authorization"] = {"static_config_approved": False, "cli_i_approve_gpu_cost": True,
                            "authorized_by": "cli_flag", "note": ""}
    assert _compare(rec)["authorization"]["authorized_by"] == "cli_flag"


def test_report_renders_roles_and_authorization(tmp_path):
    import report_benchmarks
    from compare_benchmarks import compare_renderer
    res = legacy_result()
    src = tmp_path / "E-T.json"
    src.write_text(json.dumps(res))
    cmp = compare_renderer(res)
    cmp.update({"source": str(src), "config_hash": res["config_hash"], "environment": res["environment"]})
    cp = tmp_path / "E-T.comparison.json"
    cp.write_text(json.dumps(cmp))
    assert report_benchmarks.main([str(cp)]) == 0
    md = (tmp_path / "E-T.report.md").read_text()
    assert "### Authorization" in md and "not_recorded" in md
    assert "| N=100000 512px | diagnostic |" in md and "| N=16 32px | acceptance_gate |" in md
    assert "diagnostic beyond tolerance (not a failure)" in md
    assert "| fail" not in md


@pytest.mark.skipif(not E_R1.exists(), reason="recorded E-R1 result not present")
def test_recorded_e_r1_reclassified_without_modification():
    digest = hashlib.sha256(E_R1.read_bytes()).hexdigest()
    res = json.loads(E_R1.read_text())
    cmp = _compare(res)
    assert hashlib.sha256(E_R1.read_bytes()).hexdigest() == digest
    assert cmp["gate_summary"]["acceptance_gates"] == {"pass": 16, "fail": 0, "rows": []}
    big = next(r for r in cmp["rows"] if isinstance(r["case"], dict) and r["case"]["n"] == 100000
               and r["metric"] == "forward_rgb_max_abs_err")
    assert big["baseline"] == big["candidate"] == 0.0010456162494575927
    assert big["status"].startswith("diagnostic")
    assert not any(r["status"] == "fail" for r in cmp["rows"] if r["group"] == "correctness")
    assert cmp["authorization"]["static_config_approved"] is False


# ------------------------------------------------------------------------------------------------ runner + bench
def test_run_authorization_records_cli_flag_separately():
    from run_benchmarks import run_authorization
    exp = {"id": "E", "approved": False}
    a = run_authorization(exp, True)
    assert a["static_config_approved"] is False and a["cli_i_approve_gpu_cost"] is True
    assert a["authorized_by"] == "cli_flag" and exp["approved"] is False
    assert run_authorization(exp, False)["authorized_by"] is None
    assert run_authorization({"approved": True}, False)["authorized_by"] == "static_config"


def test_runner_refuses_unapproved_without_flag(tmp_path, capsys):
    from run_benchmarks import main
    exp = tmp_path / "e.json"
    out = tmp_path / "never.json"
    exp.write_text(json.dumps({"id": "E-X", "benchmark": "renderer", "protocol": "nope.json", "approved": False,
                               "requires": [], "output": str(out)}))
    assert main(["--experiment", str(exp)]) == 3
    assert "Nothing was run" in capsys.readouterr().out and not out.exists()


def test_bench_case_classifies_and_compares_backends_without_grad_scalar_warning():
    from benchmarks.renderer.bench_renderer import run_case
    proto = {"warmup": 0, "repeats": 1, "repeat_determinism": 1}
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*requires_grad=True to a scalar.*")
        gate = run_case({"n": 8, "res": 16, "seed": 1, "gate": True}, ["reference_cpu", "reference_fp64"], proto,
                        "reference_cpu")
        diag = run_case({"n": 30, "res": 32, "seed": 2}, ["reference_cpu", "reference_fp64"], proto, "reference_cpu")
    assert gate["oracle_role"] == "acceptance_gate" and diag["oracle_role"] == "diagnostic"
    g = {x["name"]: x for x in gate["backends"]["reference_cpu"]["measurements"]}
    d = {x["name"]: x for x in diag["backends"]["reference_cpu"]["measurements"]}
    assert isinstance(g["forward_rgb_max_abs_err"]["passed"], bool)
    assert d["forward_rgb_max_abs_err"]["passed"] is None
    assert isinstance(d["forward_rgb_max_abs_err"]["within_tolerance"], bool)
    xb = {x["name"]: x for x in diag["cross_backend"]["reference_fp64"]["measurements"]}
    assert set(xb) >= {"forward_rgb_max_abs_diff_vs_reference_cpu", "grad_means_max_abs_diff_vs_reference_cpu"}
    fwd_rgb = xb["forward_rgb_max_abs_diff_vs_reference_cpu"]
    assert fwd_rgb["bitwise_equal"] is False and 0 < fwd_rgb["value"] < 1e-3       # fp64 vs fp32, same rules
