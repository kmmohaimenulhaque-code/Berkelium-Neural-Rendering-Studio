"""E-C2 Studio backend with a stub model: text -> CadQuery -> sandbox -> STEP/GLB -> core -> mesh, incl. repair."""

import importlib.util

import numpy as np
import pytest

def _real_cadquery() -> bool:          # the repo-root `cadquery/` folder can satisfy find_spec("cadquery") on its own
    try:
        return importlib.util.find_spec("cadquery.occ_impl") is not None
    except ModuleNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _real_cadquery() or
                                importlib.util.find_spec("berkelium") is None,
                                reason="needs cadquery and the Berkelium core")

GOOD = ('{"summary": "plate with a hole", "units": "mm", "assumptions": ["thickness 5 mm"], '
        '"open_questions": []}\n```python\nimport cadquery as cq\n'
        'result = cq.Workplane("XY").box(40, 20, 5).faces(">Z").workplane().hole(6)\n```')
BROKEN = GOOD.replace("result = cq.Workplane", "result = cq.Workplan")


class Stub:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, batch):
        self.calls.append(batch)
        return [self.answers.pop(0) for _ in batch]


def test_cadquery_generator_with_repair():
    from berkelium_neural.cad.cadquery_backend import CadQueryGenerator
    stub = Stub([BROKEN, GOOD])
    out = CadQueryGenerator(chat=stub).generate("a 40 x 20 x 5 mm plate with a 6 mm hole")
    rep = out["report"]
    assert len(stub.calls) == 2 and "AttributeError" in stub.calls[1][0][-1]["content"]
    assert rep["valid"] and rep["validation"]["summary"] == "pass" and len(rep["attempts"]) == 2
    assert rep["measurements"]["volume_mm3"] == pytest.approx(40 * 20 * 5 - np.pi * 9 * 5, rel=1e-6)
    assert rep["assumptions"] == ["thickness 5 mm"] and rep["scale_mm_per_unit"] == pytest.approx(40)
    v, f = out["vertices"], out["faces"]
    assert v.shape[1] == 3 and f.shape[1] == 3 and len(f) > 10 and np.ptp(v, 0).max() == pytest.approx(1.0)
    assert out["step"].startswith(b"ISO-10303-21")


def test_failure_after_two_repairs_is_reported_not_patched():
    from berkelium_neural.cad.cadquery_backend import CadQueryGenerator, CadQueryUnavailable
    with pytest.raises(CadQueryUnavailable, match="3 attempt"):
        CadQueryGenerator(chat=Stub([BROKEN] * 3)).generate("a plate")


def test_cad_generator_backend_switch_and_registry(monkeypatch):
    from berkelium_neural.cad.generator import CadGenerator
    from berkelium_neural.text3d.generators import get_generator
    g = CadGenerator(backend="cadquery", chat=Stub([GOOD]))
    assert g.identity["backend"] == "cadquery" and g.available()[0]
    assert g.generate("a plate")["report"]["backend"] == "cadquery"
    assert get_generator("cadquery", chat=Stub([GOOD])).available() == (True, "")
    monkeypatch.delenv("BERKELIUM_CAD_URL", raising=False)
    monkeypatch.delenv("BERKELIUM_CAD_BASE", raising=False)
    ok, why = get_generator("cadquery").available()
    assert not ok and "no compiler configured" in why
