"""Service wiring for the generator layer: admission (refused with a reason before any side effect), auto -> native on
the unchanged native path, the two-object planner as a durable job, and the Shap-E branch's provenance. The Shap-E
subprocess is stubbed here (unit test); the real one runs in tests/e2e/test_openworld_api.py."""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from berkelium_neural.service import executors as X
from berkelium_neural.service.api import NeuralService, create_app


@pytest.fixture()
def client(tiny_run, tmp_path, monkeypatch):
    monkeypatch.setenv("BERKELIUM_MODELS", json.dumps({"berkelium-textshape": {
        "checkpoint": str(tiny_run["base"]["artifact"]), "adapters": {}}}))
    with TestClient(create_app(NeuralService(tmp_path / "home"))) as c:
        yield c


def _run(c, **body):
    r = c.post("/v1/neural/jobs", json={"kind": "text_to_3d", "resolution": 24, **body})
    assert r.status_code == 202, r.json()
    jid, end = r.json()["id"], time.time() + 120
    while time.time() < end:
        j = c.get(f"/v1/neural/jobs/{jid}").json()
        if j["state"] in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def _mesh(c, j):
    recs = [c.get(f"/v1/neural/artifacts/{a}").json() for a in j["artifacts"]]
    return next(r for r in recs if r["kind"] == "triangle_mesh"), recs


def test_capabilities_list_generators_with_reasons(client):
    g = client.get("/v1/neural/capabilities").json()["generators"]
    assert set(g) == {"berkelium-textshape", "shap-e-text300m", "zimage-triposr", "cad", "cadquery", "auto"}
    assert g["berkelium-textshape"]["available"] and g["auto"]["available"]
    assert g["zimage-triposr"]["available"] is False and g["zimage-triposr"]["reason"].startswith("GPU-gated")


def test_unavailable_generators_are_refused_at_submission(client, tmp_path, monkeypatch):
    r = client.post("/v1/neural/jobs", json={"kind": "text_to_3d", "prompt": "a teapot", "generator": "zimage-triposr"})
    assert r.status_code == 503 and "GPU-gated" in r.json()["detail"]["message"]
    from berkelium_neural.text3d import shap_e_backend
    monkeypatch.setattr(shap_e_backend, "DEFAULT_CACHE", tmp_path / "empty")
    for body in ({"prompt": "a teapot", "generator": "shap-e-text300m"}, {"prompt": "a teapot", "generator": "auto"}):
        r = client.post("/v1/neural/jobs", json={"kind": "text_to_3d", **body})
        assert r.status_code == 503 and "weights missing" in r.json()["detail"]["message"], body
    assert client.get("/v1/neural/jobs").json() == []                   # nothing was queued


def test_auto_in_vocab_prompt_uses_the_native_path(client):
    j = _run(client, prompt="a red sphere", generator="auto")
    assert j["state"] == "completed", j.get("error")
    mesh, recs = _mesh(client, j)
    p = mesh["producer"]
    assert p["generator"] == "berkelium-textshape" and p["pipeline"] == "text->SDF->marching_cubes"
    assert p["routing"][0]["generator"] == "berkelium-textshape" and "base_checkpoint_sha256" in p
    assert {r["kind"] for r in recs} == {"triangle_mesh", "gaussian_scene", "image"}


def test_planner_job_stores_separately_validated_parts(client):
    j = _run(client, prompt="a red sphere next to a blue box", generator="berkelium-textshape", compose=True)
    assert j["state"] == "completed", j.get("error")
    mesh, _ = _mesh(client, j)
    names = {f["name"] for f in mesh["files"]}
    assert {"part_left.glb", "part_right.glb", "parts.json", "mesh.glb"} <= names
    parts = json.loads(client.get(f"/v1/neural/artifacts/{mesh['id']}/files/parts.json").content)
    assert parts["plan"]["relation"] == "next_to" and [p["prompt"] for p in parts["parts"]] == ["a red sphere",
                                                                                                 "a blue box"]
    assert all(p["report"]["valid"] for p in parts["parts"]) and len(parts["parts"][0]["transform"]) == 4
    assert mesh["producer"]["planner"]["relation"] == "next_to" and mesh["validation"]["planned"] is True


def test_shap_e_branch_records_provenance_and_label(client, monkeypatch):
    calls = []

    def fake_generate(self, prompt, seed=0, cancel=None, progress=None):
        import trimesh
        calls.append((prompt, seed, self.steps))
        progress(0.5, "sampling")
        m = trimesh.creation.icosphere(subdivisions=2, radius=0.8)
        return {"vertices": np.asarray(m.vertices, np.float32), "faces": np.asarray(m.faces),
                "colors": np.full((len(m.vertices), 3), 0.4, np.float32),
                "report": {"valid": True, "seconds": 1.5, "memory_mib": {"peak_VmRSS": 2100}}}
    monkeypatch.setattr(X.SubprocessShapE, "generate", fake_generate)
    monkeypatch.setattr(X.SubprocessShapE, "available", lambda self: (True, ""))
    monkeypatch.setattr(X, "_generator_available", lambda name: (True, ""))
    j = _run(client, prompt="a teapot", generator="auto", seed=5, shap_e_steps=8)
    assert j["state"] == "completed", j.get("error")
    mesh, recs = _mesh(client, j)
    p = mesh["producer"]
    assert calls == [("a teapot", 5, 8)]
    assert p["generator"] == "shap-e-text300m" and p["requested_generator"] == "auto"
    ident = p["generator_identity"]["shap-e-text300m"]
    assert ident["weights_sha256"]["text_cond.pt"].startswith("e6b4fa59") and ident["sampler"]["karras_steps"] == 8
    assert p["routing"][0]["unknown_words"] == ["teapot"] and p["seed"] == 5 and p["peak_rss_mib"] == 2100
    assert mesh["labels"]["provenance"] == "model-generated, normalized units, not measured geometry"
    assert {r["kind"] for r in recs} == {"triangle_mesh", "gaussian_scene", "image"}


def test_shap_e_worker_is_not_started_without_enough_free_memory(tmp_path, monkeypatch):
    from berkelium_neural.text3d import shap_e_worker
    monkeypatch.setattr(shap_e_worker, "mem_available_mib", lambda: 900)
    started = []
    monkeypatch.setattr(X.subprocess, "Popen", lambda *a, **k: started.append(a) or pytest.fail("worker started"))
    with pytest.raises(RuntimeError, match="RESOURCE_LIMIT: 900 MiB"):
        X.SubprocessShapE(16, tmp_path).generate("a teapot")
    assert started == []


def test_cad_job_runs_core_geometry_through_gaussians_and_render(client, monkeypatch):
    """Studio path for the E-C1 compiler: text -> (stub) compiler -> DesignProposal -> Berkelium core -> mesh ->
    Gaussians -> splat render, with the proposal and the core's validation stored next to the mesh."""
    pytest.importorskip("berkelium", reason="Berkelium core not installed (configs/environments/berkelium-core.txt)")
    from berkelium_neural.cad import families as FAM
    from berkelium_neural.cad.generator import CadGenerator
    from berkelium_neural.text3d import generators as G
    text, prop = FAM.sample("l_bracket", 4, False)
    real = G.get_generator
    monkeypatch.setattr(G, "get_generator", lambda n, **kw: CadGenerator(compiler=lambda t: json.dumps(prop))
                        if n == "cad" else real(n, **kw))
    j = _run(client, prompt=text, generator="cad")
    assert j["state"] == "completed", j.get("error")
    mesh, recs = _mesh(client, j)
    assert {r["kind"] for r in recs} == {"triangle_mesh", "gaussian_scene", "image"}
    names = {f["name"] for f in mesh["files"]}
    assert {"mesh.glb", "design_proposal.json", "core_validation.json"} <= names
    assert mesh["producer"]["generator"] == "cad" and "Berkelium core" in mesh["producer"]["pipeline"]
    assert mesh["labels"]["provenance"].startswith("design proposed by the E-C1 compiler")
