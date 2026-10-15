"""End-to-end, real inference: an open-world prompt through the real backend (services/backend/main.py, /v1/neural)
-> durable text_to_3d job with generator=auto -> routed to Shap-E (the native model cannot read the prompt) -> Shap-E
worker subprocess on CPU -> validated triangle-mesh artifact with provenance -> Gaussian scene -> splat render ->
an explicit-camera render job. Then a second Shap-E job is cancelled mid-sampling and its worker must be gone.

No mocks. Takes ~10-15 min on a 2-vCPU host (16 Karras steps, the P2 cost model). Writes results/e2e/
openworld_api_receipt.json.   python -m pytest tests/e2e/test_openworld_api.py -q
"""

import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "e2e"
PROMPT = "a wooden chair with four legs"
STEPS = 16


def _port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _req(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                               method="POST" if body is not None else "GET")
    with urllib.request.urlopen(r, timeout=60) as resp:
        raw = resp.read()
        return json.loads(raw) if resp.headers.get_content_type() == "application/json" else raw


def _workers():
    out = subprocess.run(["pgrep", "-f", "berkelium_neural.text3d.shap_e_worker"], capture_output=True, text=True)
    return [p for p in out.stdout.split() if p and int(p) != os.getpid()]


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "services" / "neural"))
    from berkelium_neural.text3d.shap_e_backend import available
    ok, why = available()
    if not ok:
        pytest.skip(f"Shap-E unavailable: {why} (scripts/setup_openworld.sh)")
    tmp = tmp_path_factory.mktemp("e2e_openworld")
    port = _port()
    cfg = tmp / "config.json"
    cfg.write_text(json.dumps({"backend_api_key": "", "allowed_origins": ["http://127.0.0.1:5173"]}))
    env = {**os.environ, "BERKELIUM_CONFIG": str(cfg), "BERKELIUM_NEURAL_HOME": str(tmp / "neural")}
    log = open(tmp / "backend.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=ROOT / "services" / "backend", env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    end = time.time() + 90
    while time.time() < end:
        try:
            _req(f"{url}/health")
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    try:
        yield url
    finally:
        proc.terminate()
        proc.wait(30)
        log.close()


def _wait(url, jid, timeout, stages):
    end = time.time() + timeout
    while time.time() < end:
        j = _req(f"{url}/v1/neural/jobs/{jid}")
        if j.get("stage") and (not stages or stages[-1] != j["stage"]):
            stages.append(j["stage"])
        if j["state"] in ("completed", "failed", "cancelled"):
            return j
        time.sleep(2)
    raise TimeoutError(f"job {jid} not finished after {timeout}s (stages {stages[-5:]})")


def test_open_world_prompt_end_to_end_with_real_shap_e(backend):
    import numpy as np
    from PIL import Image
    caps = _req(f"{backend}/v1/neural/capabilities")
    assert caps["generators"]["shap-e-text300m"]["available"], caps["generators"]["shap-e-text300m"]
    t0 = time.time()
    job = _req(f"{backend}/v1/neural/jobs", {"kind": "text_to_3d", "prompt": PROMPT, "generator": "auto",
                                             "shap_e_steps": STEPS, "seed": 0})
    stages = []
    j = _wait(backend, job["id"], 2400, stages)
    wall = time.time() - t0
    assert j["state"] == "completed", j.get("error")
    recs = {r["kind"]: r for r in (_req(f"{backend}/v1/neural/artifacts/{a}") for a in j["artifacts"])}
    mesh, gauss, img = recs["triangle_mesh"], recs["gaussian_scene"], recs["image"]
    p = mesh["producer"]
    assert p["generator"] == "shap-e-text300m" and p["requested_generator"] == "auto"
    assert p["routing"][0]["unknown_words"] == ["chair", "four", "legs", "wooden"]
    ident = p["generator_identity"]["shap-e-text300m"]
    assert ident["shap_e_commit"] == "50131012ee11c9d2617f3886c10f000d3c7a3b43"
    assert ident["sampler"]["karras_steps"] == STEPS and ident["execution"].startswith("subprocess")
    assert mesh["labels"]["provenance"] == "model-generated, normalized units, not measured geometry"
    assert mesh["validation"]["valid"] and mesh["validation"]["n_faces"] > 100
    assert any("sampling" in s for s in stages) and any("decoding" in s for s in stages), stages
    assert gauss["parents"] == [mesh["id"]] and img["parents"] == [gauss["id"]]
    png = _req(f"{backend}/v1/neural/artifacts/{img['id']}/files/render.png")
    im = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.float32)
    assert im.std() > 5, "splat render is blank"
    cam = json.loads(_req(f"{backend}/v1/neural/artifacts/{img['id']}/files/cameras.json"))[1]
    rj = _req(f"{backend}/v1/neural/jobs", {"kind": "render", "artifact_id": gauss["id"], "camera": cam,
                                            "backend": "reference_cpu", "config": {"background": [1, 1, 1]}})
    r = _wait(backend, rj["id"], 600, [])
    assert r["state"] == "completed", r.get("error")
    assert _workers() == []                                    # the worker exited with its job
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "openworld_api_mesh.glb").write_bytes(_req(f"{backend}/v1/neural/artifacts/{mesh['id']}/files/mesh.glb"))
    (OUT / "openworld_api_render.png").write_bytes(png)
    receipt = {"prompt": PROMPT, "job": j["id"], "wall_s": round(wall, 1), "stages_seen": stages,
               "mesh": {"id": mesh["id"], "n_faces": mesh["validation"]["n_faces"],
                        "components": mesh["validation"]["components"], "watertight": mesh["validation"]["watertight"]},
               "producer": {k: p[k] for k in ("generator", "requested_generator", "seed", "seconds", "peak_rss_mib",
                                              "routing", "pipeline")},
               "gaussian_scene": {"id": gauss["id"], "n": gauss["validation"]["n"]},
               "render": {"image": img["id"], "renderer": img["producer"]["renderer"], "std": float(im.std()),
                          "explicit_camera_job": r["id"]},
               "device": "cpu", "training_performed": "none"}
    (OUT / "openworld_api_receipt.json").write_text(json.dumps(receipt, indent=2, default=str))


def test_cancelling_a_shap_e_job_stops_its_worker(backend):
    job = _req(f"{backend}/v1/neural/jobs", {"kind": "text_to_3d", "prompt": "a teapot", "generator": "shap-e-text300m",
                                             "shap_e_steps": STEPS})
    end = time.time() + 600
    while time.time() < end:
        j = _req(f"{backend}/v1/neural/jobs/{job['id']}")
        if "sampling" in (j.get("stage") or "") or j["state"] in ("failed", "completed"):
            break
        time.sleep(1)
    assert "sampling" in (j.get("stage") or ""), j
    _req(f"{backend}/v1/neural/jobs/{job['id']}/cancel", {})
    j = _wait(backend, job["id"], 60, [])
    assert j["state"] == "cancelled", j
    time.sleep(2)
    assert _workers() == []
