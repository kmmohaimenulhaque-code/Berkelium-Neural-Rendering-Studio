"""FastAPI job/artifact interface: validation before side effects, real text_to_3d + render jobs, honest backend
failures, admission control, cancellation, restart recovery, artifact immutability and corruption detection."""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from berkelium_neural.camera import make_camera
from berkelium_neural.contracts import JobState
from berkelium_neural.service.api import NeuralService, create_app
from berkelium_neural.service.artifacts import ArtifactError, ArtifactStore
from berkelium_neural.service.jobs import JobManager


@pytest.fixture()
def client(tiny_run, tmp_path, monkeypatch):
    monkeypatch.setenv("BERKELIUM_MODELS", json.dumps({"berkelium-textshape": {
        "checkpoint": str(tiny_run["base"]["artifact"]), "adapters": {"capsule-v1": str(tiny_run["lora"]["artifact"])}}}))
    svc = NeuralService(tmp_path / "home", max_pending=3)
    with TestClient(create_app(svc)) as c:
        c.svc = svc
        yield c


def wait(c, jid, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/v1/neural/jobs/{jid}").json()
        if j["state"] in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_text_to_3d_job_produces_mesh_and_gaussian_artifacts(client):
    r = client.post("/v1/neural/jobs", json={"kind": "text_to_3d", "prompt": "a red sphere", "resolution": 24})
    assert r.status_code == 202
    j = wait(client, r.json()["id"])
    assert j["state"] == "completed", j.get("error")
    states = [t["to"] for t in j["transitions"]]
    assert states[0] == "pending" and states[-1] == "completed" and "running" in states
    mesh_id, gauss_id = j["artifacts"]
    mesh = client.get(f"/v1/neural/artifacts/{mesh_id}").json()
    assert mesh["kind"] == "triangle_mesh" and mesh["labels"]["not"].startswith("CAD")
    assert mesh["producer"]["base_checkpoint_sha256"]
    glb = client.get(f"/v1/neural/artifacts/{mesh_id}/files/mesh.glb")
    assert glb.status_code == 200 and glb.headers["content-type"] == "model/gltf-binary"
    import trimesh
    m = trimesh.load(io.BytesIO(glb.content), file_type="glb", force="mesh")
    assert len(m.faces) == mesh["validation"]["n_faces"]
    g = client.get(f"/v1/neural/artifacts/{gauss_id}").json()
    assert g["kind"] == "gaussian_scene" and g["parents"] == [mesh_id]


def _gaussian_artifact_and_camera(client):
    j = wait(client, client.post("/v1/neural/jobs", json={"kind": "text_to_3d", "prompt": "a blue box",
                                                           "resolution": 24}).json()["id"])
    cam = json.loads(make_camera((0, 0.5, 3.2), width=32, height=32).model_dump_json())
    return j["artifacts"][1], cam


def _render(client, gauss_id, cam, backend):
    return wait(client, client.post("/v1/neural/jobs", json={"kind": "render", "artifact_id": gauss_id, "camera": cam,
                                                             "backend": backend}).json()["id"])


def test_render_job_reference_cpu(client):
    gauss_id, cam = _gaussian_artifact_and_camera(client)
    ok = _render(client, gauss_id, cam, "reference_cpu")
    assert ok["state"] == "completed"
    img = client.get(f"/v1/neural/artifacts/{ok['artifacts'][0]}").json()
    assert img["kind"] == "image" and img["producer"]["renderer"] == "reference_cpu"
    png = client.get(f"/v1/neural/artifacts/{ok['artifacts'][0]}/files/render.png")
    from PIL import Image
    assert len(set(Image.open(io.BytesIO(png.content)).convert("RGB").getdata())) > 1   # same scene/camera as GPU test


@pytest.mark.parametrize("unavailable,expected_message", [
    ("gsplat_not_importable", "gsplat not importable"),
    ("not_rocm_build", "ROCm PyTorch required"),
    ("no_device", "no GPU device visible"),
])
def test_render_job_gsplat_unavailable_fails_honestly(client, monkeypatch, unavailable, expected_message):
    """Simulated unavailability, independent of the machine (CPU CI or a working MI300X): only the availability
    report that GsplatRenderer checks before allocating is patched — starting from "everything available" and
    knocking out exactly one condition, so the message proves which check fired. The renderer is not mocked."""
    from berkelium_neural.render import gsplat_adapter
    real_status = gsplat_adapter.gsplat_status

    def simulated():
        st = dict(real_status())
        st.pop("gsplat_import_error", None)
        st.update(gsplat_version=st.get("gsplat_version") or "simulated", torch_hip=st["torch_hip"] or "simulated",
                  device_visible=True)
        if unavailable == "no_device":
            st["device_visible"] = False
        elif unavailable == "not_rocm_build":
            st["torch_hip"] = None
        else:
            st.pop("gsplat_version")
            st["gsplat_import_error"] = "ModuleNotFoundError: simulated"
        st["usable"] = False
        return st
    monkeypatch.setattr(gsplat_adapter, "gsplat_status", simulated)
    gauss_id, cam = _gaussian_artifact_and_camera(client)
    bad = _render(client, gauss_id, cam, "gsplat")
    assert bad["state"] == "failed" and bad["error"]["code"] == "DEVICE_UNAVAILABLE" and not bad["artifacts"]
    assert expected_message in bad["error"]["message"], bad["error"]


def test_render_job_gsplat_fails_honestly_when_actually_unavailable(client):
    """Real (unpatched) availability: on a machine without ROCm gsplat the job must fail, never fake an image."""
    from berkelium_neural.render.gsplat_adapter import gsplat_status
    if gsplat_status()["usable"]:
        pytest.skip("gsplat is usable here; covered by test_render_job_gsplat_real_render")
    gauss_id, cam = _gaussian_artifact_and_camera(client)
    bad = _render(client, gauss_id, cam, "gsplat")
    assert bad["state"] == "failed" and bad["error"]["code"] == "DEVICE_UNAVAILABLE" and not bad["artifacts"]


@pytest.mark.hip
@pytest.mark.gpu
def test_render_job_gsplat_real_render(client):
    """Success path with the real AMD gsplat backend (no mocks). Runs only where gsplat is actually usable."""
    from berkelium_neural.render.gsplat_adapter import gsplat_status
    st = gsplat_status()
    if not st["usable"]:
        pytest.skip(f"requires ROCm PyTorch + visible HIP device + gsplat (status: usable=False, "
                    f"hip={st['torch_hip']}, device={st['device_visible']}, "
                    f"import_error={st.get('gsplat_import_error')})")
    gauss_id, cam = _gaussian_artifact_and_camera(client)
    ok = _render(client, gauss_id, cam, "gsplat")
    assert ok["state"] == "completed", ok.get("error")
    assert len(ok["artifacts"]) == 1
    img = client.get(f"/v1/neural/artifacts/{ok['artifacts'][0]}").json()
    assert img["kind"] == "image" and img["producer"]["renderer"] == "gsplat"
    assert img["producer"]["backend"] == "gsplat" and img["producer"]["profile"] == "gsplat_v1"
    png = client.get(f"/v1/neural/artifacts/{ok['artifacts'][0]}/files/render.png")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    im = Image.open(io.BytesIO(png.content))
    assert im.size == (32, 32)
    assert len(set(im.convert("RGB").getdata())) > 1, "render is a constant image"


@pytest.mark.parametrize("body,status", [
    ({"kind": "text_to_3d", "prompt": "   "}, 422),
    ({"kind": "text_to_3d", "prompt": "a\x00b"}, 422),
    ({"kind": "text_to_3d", "prompt": "x" * 301}, 422),
    ({"kind": "text_to_3d", "prompt": "ok", "resolution": 4096}, 422),
    ({"kind": "shell", "cmd": "rm -rf /"}, 422),
    ({"kind": "text_to_3d", "prompt": "ok", "model": "not-a-model"}, 503),
    ({"kind": "text_to_3d", "prompt": "ok", "adapter": "nope"}, 503),
    ({"kind": "render", "artifact_id": "0" * 64, "camera": {}}, 422),
])
def test_invalid_requests_rejected_before_side_effects(client, body, status):
    before = len(list((client.svc.root / "jobs").glob("*")))
    r = client.post("/v1/neural/jobs", json=body)
    assert r.status_code == status, r.text
    assert client.get("/v1/neural/jobs").json() == []
    assert len(list((client.svc.root / "jobs").glob("*"))) == before


def test_path_and_id_validation(client):
    assert client.get("/v1/neural/artifacts/..%2f..%2fetc").status_code in (404, 422)
    assert client.get("/v1/neural/artifacts/" + "a" * 64).status_code == 404
    assert client.get("/v1/neural/jobs/../../x").status_code in (404, 422)


def test_admission_limit_and_cancel(tiny_run, tmp_path):
    gate = __import__("threading").Event()

    def slow(req, ctx):
        while not gate.is_set():
            ctx.check()
            time.sleep(0.01)
    m = JobManager(tmp_path / "j", {"text_to_3d": slow}, max_pending=2)
    a = m.submit({"kind": "text_to_3d", "prompt": "x"})
    b = m.submit({"kind": "text_to_3d", "prompt": "y"})
    with pytest.raises(Exception) as e:
        m.submit({"kind": "text_to_3d", "prompt": "z"})
    assert getattr(e.value, "code", "") == "RESOURCE_LIMIT"
    assert m.cancel(b.id).state == JobState.CANCELLED                 # pending -> cancelled immediately
    m.cancel(a.id)                                                     # running -> cooperative cancel
    assert m.wait(a.id).state == JobState.CANCELLED
    gate.set()
    m.shutdown()


def test_restart_recovery_resumes_interrupted_jobs(tmp_path):
    """A job left RUNNING by a crashed process is resumed (not silently completed) by the next manager."""
    import sqlite3
    runs = []

    def ok(req, ctx):
        runs.append(req["prompt"])
    JobManager(tmp_path / "j", {"text_to_3d": ok}).shutdown()          # creates the database
    db = sqlite3.connect(tmp_path / "j" / "jobs.sqlite3")
    jid = "c" * 32
    db.execute("INSERT INTO jobs VALUES (?,?,?,0.3,'generate_sdf','[]',NULL,'t','t')",
               (jid, json.dumps({"kind": "text_to_3d", "prompt": "after-crash"}), JobState.RUNNING.value))
    db.commit()
    db.close()
    m2 = JobManager(tmp_path / "j", {"text_to_3d": ok})
    j = m2.wait(jid)
    assert j.state == JobState.COMPLETED and runs == ["after-crash"]
    tos = [t["to"] for t in m2.transitions(jid)]
    assert tos[:2] == ["resumed", "running"] and tos[-1] == "completed"
    m2.shutdown()


def test_artifact_store_immutability_and_corruption(tmp_path):
    s = ArtifactStore(tmp_path / "a")
    r1 = s.create("report", {"x.json": (b'{"a":1}', "application/json")}, {"p": 1})
    r2 = s.create("report", {"x.json": (b'{"a":1}', "application/json")}, {"p": 1})
    assert r1.id == r2.id and r1.created_at == r2.created_at          # dedupe, never overwrite
    blob = s._blob(r1.files[0].sha256)
    blob.write_bytes(b'{"a":2}')
    with pytest.raises(ArtifactError) as e:
        s.read_file(r1.id, "x.json")
    assert e.value.code == "CORRUPTED"
    with pytest.raises(ArtifactError):
        s.create("report", {"y": (b"", "text/plain")}, {}, parents=("f" * 64,))
