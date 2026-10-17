import importlib.util
import os

import pytest

import sandbox
from unsafe_corpus import CANARY, UNSAFE

needs_cq = pytest.mark.skipif(importlib.util.find_spec("cadquery") is None, reason="cadquery not installed")

GOOD = "import cadquery as cq\nimport math\nresult = cq.Workplane('XY').box(10, 20, 30).faces('>Z').hole(4)\n"


@pytest.fixture(scope="module")
def runner():
    with sandbox.BatchRunner(jobs=4, timeout=10, keep_artifacts=True) as r:
        yield r


def test_static_check_accepts_plain_cadquery():
    assert sandbox.check_script(GOOD) == []
    assert sandbox.check_script("from math import pi, sqrt\nimport cadquery as cq\nresult = 1\n") == []


@pytest.mark.parametrize("name", sorted(UNSAFE))
def test_static_check_or_runtime_rejects(name):
    # every corpus entry is rejected statically except the three resource bombs and the no-result script,
    # which are stopped at run time (covered below)
    runtime_only = {"infinite_loop", "memory_bomb", "cpu_bomb"}
    bad = sandbox.check_script(UNSAFE[name])
    assert bool(bad) != (name in runtime_only), (name, bad)


@needs_cq
def test_good_script_runs(runner):
    r = runner.run(GOOD)
    assert r.ok and r.stage == "ok" and r.valid and r.n_solids == 1
    assert r.volume == pytest.approx(10 * 20 * 30 - 3.141592653589793 * 4 * 30, rel=1e-6)
    assert r.bbox == pytest.approx([-5, -10, -15, 5, 10, 15])
    assert r.step.startswith(b"ISO-10303-21") and r.glb[:4] == b"glTF"


@needs_cq
def test_single_shot_runner_matches_batch(runner):
    a, b = sandbox.run_script(GOOD, timeout=30), runner.run(GOOD)
    assert a.ok and a.volume == pytest.approx(b.volume)


@needs_cq
def test_all_unsafe_scripts_rejected_and_no_side_effect(runner):
    if os.path.exists(CANARY):
        os.remove(CANARY)
    results = dict(zip(sorted(UNSAFE), runner.map([UNSAFE[k] for k in sorted(UNSAFE)])))
    accepted = [k for k, r in results.items() if r.ok]
    assert accepted == []
    assert not os.path.exists(CANARY) and not os.path.exists(CANARY + ".step")
    assert results["infinite_loop"].stage == "timeout"
    assert results["memory_bomb"].stage in ("memory", "runtime")


@needs_cq
def test_runtime_errors_are_reported(runner):
    r = runner.run("import cadquery as cq\nresult = cq.Workplane().box(1, 1, 1).faces('>Z').fillet(5)\n")
    assert not r.ok and r.stage in ("runtime", "invalid") and r.error
    r = runner.run("import cadquery as cq\nresult = 42\n")
    assert r.stage == "no_result"


@needs_cq
def test_environment_is_empty(runner):
    os.environ["HF_TOKEN"] = "secret-should-not-leak"
    try:
        # the static check forbids os/env access outright; this proves the child env is empty as well
        r = sandbox.run_script("import cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)\n", timeout=30)
        assert r.ok
        assert "HF_TOKEN" not in sandbox._child_env("/tmp")
    finally:
        del os.environ["HF_TOKEN"]
