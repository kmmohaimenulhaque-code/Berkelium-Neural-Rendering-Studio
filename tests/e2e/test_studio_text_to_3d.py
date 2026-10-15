"""End-to-end: Studio (built React app in headless Chromium, WebGL via SwiftShader) -> the Berkelium backend
(services/backend/main.py with /v1/neural) -> real text_to_3d job with the released checkpoint -> mesh imported into
the scene -> Gaussian scene splat-rendered by the real renderer -> "Render this view" from the live viewport camera ->
backend restart -> the job and its artifacts are still listed and retrievable in Studio.

Asserts on identities, not appearances: the model in the scene has the triangle count of the backend's validated
mesh artifact, the displayed images are the backend's image artifacts (parented to the Gaussian scene), and the
rendered images are not blank. Writes screenshots and a receipt to results/e2e/.

  python -m pytest tests/e2e -q        # needs apps/desktop/node_modules and `playwright install chromium`
"""

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "apps" / "desktop"
OUT = ROOT / "results" / "e2e"
PROMPT = "a red cone on top of a blue box"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.status
        except Exception:  # noqa: BLE001
            time.sleep(0.25)
    raise TimeoutError(url)


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


class Backend:
    """The real backend process; restartable on the same neural home (durable jobs + artifacts)."""

    def __init__(self, port, env):
        self.port, self.env, self.proc = port, env, None
        self.url = f"http://127.0.0.1:{port}"

    def start(self):
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port",
                                      str(self.port)], cwd=ROOT / "services" / "backend", env=self.env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        _wait_http(f"{self.url}/health")

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    if not (APP / "node_modules").exists():
        pytest.skip("apps/desktop/node_modules missing: run `npm ci` in apps/desktop")
    tmp = tmp_path_factory.mktemp("e2e")
    api_port, ui_port = _free_port(), _free_port()
    cfg = tmp / "config.json"
    cfg.write_text(json.dumps({"backend_api_key": "", "allowed_origins": [f"http://127.0.0.1:{ui_port}"]}))
    env = {**os.environ, "BERKELIUM_CONFIG": str(cfg), "BERKELIUM_NEURAL_HOME": str(tmp / "neural")}
    if os.environ.get("BERKELIUM_E2E_MODEL"):        # default: the released checkpoint in exports/
        env["BERKELIUM_MODELS"] = json.dumps({"berkelium-textshape": {
            "checkpoint": os.environ["BERKELIUM_E2E_MODEL"], "adapters": {}}})
    api = Backend(api_port, env)
    api.start()
    dist = tmp / "dist"
    b = subprocess.run(["npx", "vite", "build", "--outDir", str(dist), "--emptyOutDir"], cwd=APP, capture_output=True,
                       text=True, check=False, env={**os.environ, "VITE_BERKELIUM_API": api.url})
    assert b.returncode == 0, b.stdout + b.stderr
    ui = subprocess.Popen(["npx", "vite", "preview", "--outDir", str(dist), "--host", "127.0.0.1", "--port",
                           str(ui_port), "--strictPort"], cwd=APP, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        _wait_http(f"http://127.0.0.1:{ui_port}/")
        yield {"api": api, "ui": f"http://127.0.0.1:{ui_port}"}
    finally:
        ui.terminate()
        api.stop()


def _open_scene(page, ui):
    page.goto(ui)
    page.get_by_role("tab", name="Scene").click()
    page.wait_for_function("window.__berkeliumNeural && window.__berkeliumNeural.ready", timeout=30000)
    page.get_by_test_id("generate-btn").wait_for(state="visible", timeout=30000)
    page.wait_for_function("!document.querySelector('[data-testid=generate-btn]').disabled", timeout=30000)


def _png_stats(data: bytes):
    import numpy as np
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return im.size, float(np.asarray(im, dtype=np.float32).std())


def test_prompt_to_rendered_result_in_studio_and_after_restart(stack):
    from playwright.sync_api import sync_playwright
    api = stack["api"]
    health = json.loads(_get(f"{api.url}/health"))
    assert health["components"]["neural"]["status"] == "ok", health
    assert all(health["components"]["neural"]["models"].values()), health     # released checkpoint present
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"prompt": PROMPT}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                          "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        logs = []
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        _open_scene(page, stack["ui"])

        # 1. prompt -> real model inference -> mesh imported into the scene
        page.get_by_test_id("prompt-input").fill(PROMPT)
        page.get_by_test_id("generate-btn").click()
        page.get_by_test_id("job-state").filter(has_text="completed").wait_for(timeout=240000)
        page.get_by_test_id("render-image").wait_for(state="visible", timeout=60000)
        st = page.evaluate("window.__berkeliumNeural")
        mesh = json.loads(_get(f"{api.url}/v1/neural/artifacts/{st['meshArtifact']}"))
        assert mesh["kind"] == "triangle_mesh" and mesh["validation"]["valid"]
        assert mesh["producer"]["base_checkpoint_sha256"] == json.loads(
            (ROOT / "exports" / "textshape-base-v1" / "MODEL_CARD.json").read_text())["files"]["model.pt"] \
            or os.environ.get("BERKELIUM_E2E_MODEL")
        props = page.locator(".props").inner_text()       # the scene's Model panel for the selected, imported node
        assert "text-to-3D" in props and mesh["id"][:12] in props, props
        assert f"{mesh['validation']['n_faces']:,}" in props, (props, mesh["validation"]["n_faces"])

        # 2. the Gaussian scene built from the mesh, splat-rendered by the real renderer, shown in Studio
        gauss = json.loads(_get(f"{api.url}/v1/neural/artifacts/{st['gaussianArtifact']}"))
        assert gauss["kind"] == "gaussian_scene" and gauss["parents"] == [mesh["id"]]
        img = json.loads(_get(f"{api.url}/v1/neural/artifacts/{st['renderArtifact']}"))
        assert img["kind"] == "image" and img["parents"] == [gauss["id"]]
        assert img["producer"]["pipeline"] == "gaussian_scene->splat_render"
        size, std = _png_stats(_get(f"{api.url}/v1/neural/artifacts/{img['id']}/files/render.png"))
        assert size == (512, 512) and std > 5.0, (size, std)
        page.screenshot(path=str(OUT / "studio_text_to_3d.png"))

        # 3. "Render this view": the live viewport camera -> CameraSpec -> render job on the Gaussian scene
        page.get_by_test_id("render-view-btn").click()
        page.wait_for_function("window.__berkeliumNeural.viewRender", timeout=120000)
        vr = page.evaluate("window.__berkeliumNeural.viewRender")
        view = json.loads(_get(f"{api.url}/v1/neural/artifacts/{vr['artifact']}"))
        assert view["kind"] == "image" and view["parents"] == [gauss["id"]]
        vdata = _get(f"{api.url}/v1/neural/artifacts/{view['id']}/files/render.png")
        vsize, vstd = _png_stats(vdata)
        assert vsize == (vr["width"], vr["height"]) and vstd > 3.0, (vsize, vstd)
        assert view["validation"]["meta"]["n_visible"] > 1000, view["validation"]["meta"]    # the object is in view
        (OUT / "view_render.png").write_bytes(vdata)
        page.screenshot(path=str(OUT / "studio_view_render.png"))
        receipt.update({"job": st["jobId"], "mesh_artifact": mesh["id"], "faces": mesh["validation"]["n_faces"],
                        "gaussian_artifact": gauss["id"], "gaussians": gauss["validation"]["n"],
                        "orbit_render_artifact": img["id"], "renderer": img["producer"]["renderer"],
                        "view_render": {**vr, "n_visible": view["validation"]["meta"]["n_visible"], "std": vstd}})

        errors_before_restart = [x for x in logs if x.startswith("error")]
        assert not errors_before_restart, errors_before_restart     # the main journey logs no browser errors

        # 4. durable jobs: restart the backend on the same home; Studio still lists and reopens the result
        api.stop()
        api.start()
        j = json.loads(_get(f"{api.url}/v1/neural/jobs/{st['jobId']}"))
        assert j["state"] == "completed" and j["artifacts"][0] == mesh["id"]
        page.reload()
        _open_scene(page, stack["ui"])
        recent = page.get_by_test_id("recent-jobs")
        recent.wait_for(timeout=30000)
        assert PROMPT in recent.inner_text()
        recent.get_by_role("button", name="open").first.click()
        page.get_by_test_id("render-image").wait_for(state="visible", timeout=60000)
        after = page.evaluate("window.__berkeliumNeural")
        assert after["meshArtifact"] == mesh["id"] and after["renderArtifact"] == img["id"]
        page.screenshot(path=str(OUT / "studio_after_restart.png"))
        browser.close()
    receipt["after_restart"] = {"job_state": j["state"], "reopened_mesh": after["meshArtifact"]}
    receipt["console_errors_before_restart"] = errors_before_restart
    receipt["console_errors_during_restart"] = len([x for x in logs if x.startswith("error")]) - len(errors_before_restart)
    (OUT / "e2e_receipt.json").write_text(json.dumps(receipt, indent=2))
    shutil.copy(OUT / "studio_text_to_3d.png", OUT / "latest.png")
