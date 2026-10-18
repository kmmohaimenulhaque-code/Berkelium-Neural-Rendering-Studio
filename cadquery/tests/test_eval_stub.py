import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("cadquery") is None or
                                importlib.util.find_spec("berkelium") is None,
                                reason="needs cadquery and the Berkelium core")
ROOT = Path(__file__).resolve().parents[2]


def load_eval():
    spec = importlib.util.spec_from_file_location("cq_eval", ROOT / "scripts" / "cq_eval.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["cq_eval"] = m
    spec.loader.exec_module(m)
    return m


class Stub:
    """Base: answers with no code. Tuned: answers the reference completion (repair returns it too)."""
    name = "stub"

    def __init__(self, items):
        self.ref = {it["text"]: it["completion"] for it in items}

    def chat(self, batch, adapter=True):
        out = []
        for msgs in batch:
            text = msgs[1]["content"]
            out.append(self.ref[text] if adapter else "I cannot do that.")
        return out


def test_eval_scores_stub_models():
    import cqformat
    import sandbox
    import transpile
    from berkelium_neural.cad import families as F
    ev = load_eval()
    items = []
    for i, fam in enumerate(["washer", "hex_nut", "knob"]):
        text, prop = F.sample(fam, 0, False)
        code = transpile.transpile(prop)
        items.append({"id": f"x{i}", "source": "families", "split": "heldout" if fam == "knob" else "test",
                      "family": fam, "kind": "gen", "text": text, "code": code,
                      "completion": cqformat.render({"summary": fam}, code)})
    stub = Stub(items)
    with sandbox.BatchRunner(jobs=2, keep_artifacts=True) as runner:
        refs = {it["id"]: r.step for it, r in zip(items, runner.map([it["code"] for it in items]))}
        base = ev.run_model(stub, items, False, runner, rounds=1)
        tuned = ev.run_model(stub, items, True, runner, rounds=1)
    ev.score(base, refs, 2)
    ev.score(tuned, refs, 2)
    mb, mt = ev.summarize(base), ev.summarize(tuned)
    assert mb["all_heldout"]["first"]["execute_valid"] == 0.0 and mb["all_heldout"]["first"]["iou_mean"] == 0.0
    assert mt["all_heldout"]["first"]["execute_valid"] == 1.0
    assert mt["all_heldout"]["first"]["iou_mean"] == pytest.approx(1.0, abs=1e-6)
    assert mt["heldout_families"]["first"]["n"] == 1
    m, lo, hi = ev.bootstrap_ci([t["first"]["iou"] - b["first"]["iou"] for t, b in zip(tuned, base)])
    assert lo > 0.9


def test_iou_partial_overlap():
    import core_check
    import sandbox
    a = sandbox.run_script("import cadquery as cq\nresult = cq.Workplane().box(10, 10, 10)\n")
    b = sandbox.run_script("import cadquery as cq\nresult = cq.Workplane().box(10, 10, 20)\n")
    assert core_check.iou(a.step, b.step, "none") == pytest.approx(0.5, rel=1e-6)
    assert core_check.validate(a.step)["l3"] is True
