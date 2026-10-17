import importlib.util

import pytest

import sandbox
import transpile

pytestmark = pytest.mark.skipif(importlib.util.find_spec("cadquery") is None or
                                importlib.util.find_spec("berkelium") is None,
                                reason="needs cadquery and the Berkelium core")

FAMS = ["plate_holes", "l_bracket", "flange", "stepped_shaft", "hex_nut", "pulley", "pillow_block",
        "pipe_elbow", "rocket_nozzle", "propeller", "mug", "vase", "truss", "knob", "hinge_leaf", "custom_part"]


def core_measure(prop):
    from berkelium.geometry.occt import OCCTBackend
    from berkelium.pipeline import run
    g = run(prop, realize=False).record.realized[0].geometry
    occ = OCCTBackend()
    m = occ.measure(occ.execute(g).bodies["body"])
    return m.volume_mm3, list(m.bbox_min) + list(m.bbox_max)


@pytest.fixture(scope="module")
def runner():
    with sandbox.BatchRunner(jobs=4) as r:
        yield r


def test_round_trip_volume_and_bbox(runner):
    from berkelium_neural.cad import families as F
    jobs = []
    for fam in FAMS:
        for seed in range(6):
            r = F.sample(fam, seed, vague=seed % 2 == 1)
            if r is not None:
                jobs.append((fam, r[1]))
                break
    srcs = [transpile.transpile(p) for _, p in jobs]
    assert all(sandbox.check_script(s) == [] for s in srcs)
    for (fam, prop), res in zip(jobs, runner.map(srcs)):
        assert res.ok, (fam, res.error)
        vol, bb = core_measure(prop)
        assert res.volume == pytest.approx(vol, rel=1e-6), fam
        assert res.bbox == pytest.approx(bb, abs=1e-3), fam


def test_parameters_become_variables():
    from berkelium_neural.cad import families as F
    src = transpile.transpile(F.sample("plate_holes", 0, False)[1])
    assert "L = " in src and "cq.Solid.makeBox(*(L, W, T)" in src and src.rstrip().endswith("result = body")


def test_unknown_op_is_an_error():
    prop = {"structure": {"components": [{"id": "c", "kind": "procedural", "geometry": {
        "units": "mm", "ops": [{"id": "a", "op": "implicit", "expr": "x"}], "outputs": {"body": "a"}}}]}}
    with pytest.raises(transpile.TranspileError):
        transpile.transpile(prop, use_core=False)
