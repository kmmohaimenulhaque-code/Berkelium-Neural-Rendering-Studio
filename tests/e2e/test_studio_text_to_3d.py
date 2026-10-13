"""End-to-end: Studio UI (built React app in headless Chromium, WebGL via SwiftShader) -> existing FastAPI backend
(services/backend/main.py with the neural router) -> real text_to_3d job -> immutable artifacts -> Three.js viewport.

Asserts the viewport displays the SAME artifact the backend produced (face count from the GLB loaded in the
browser == the backend's validated mesh report) and that the canvas is not blank. Writes a screenshot to
results/e2e/. Uses a trained checkpoint from BERKELIUM_E2E_MODEL or trains a tiny one on the fly.
"""

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
sys.path.insert(0, str(ROOT / "services" / "neural"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.status
        except Exception:  # noqa: BLE001
            time.sleep(0.25)
    raise TimeoutError(url)


@pytest.fixture(scope="module")
def model_checkpoint(tmp_path_factory):
    if os.environ.get("BERKELIUM_E2E_MODEL"):
        return Path(os.environ["BERKELIUM_E2E_MODEL"])
    trained = ROOT / "runs" / "textshape_base_v1" / "model.pt"
    if trained.exists():
        return trained
    from berkelium_neural.text3d.dataset import build
    from berkelium_neural.text3d.train import train
    sys.path.insert(0, str(ROOT / "services" / "neural" / "tests"))
    from conftest import TINY_DATASET, tiny_train_cfg
    d = tmp_path_factory.mktemp("e2e_model")
    build(TINY_DATASET, d / "data", log=lambda *a: None)
    return Path(train(tiny_train_cfg(d / "data", d / "base", steps=30), log=lambda *a: None)["artifact"])


@pytest.fixture(scope="module")
def stack(tmp_path_factory, model_checkpoint):
    if not (APP / "node_modules").exists():
        pytest.skip("apps/desktop/node_modules missing: run `npm ci` in apps/desktop")
    tmp = tmp_path_factory.mktemp("e2e")
    api_port, ui_port = _free_port(), _free_port()
    cfg = tmp / "config.json"
    cfg.write_text(json.dumps({"fireworks_api_key": "", "model_id": "", "backend_api_key": "",
                               "allowed_origins": [f"http://127.0.0.1:{ui_port}"]}))
    env = {**os.environ, "BERKELIUM_CONFIG": str(cfg), "BERKELIUM_NEURAL_HOME": str(tmp / "neural"),
           "BERKELIUM_MODELS": json.dumps({"berkelium-textshape": {"checkpoint": str(model_checkpoint),
                                                                    "adapters": {}}})}
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port",
                            str(api_port)], cwd=ROOT / "services" / "backend", env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    dist = tmp / "dist"
    b = subprocess.run(["npx", "vite", "build", "--outDir", str(dist), "--emptyOutDir"], cwd=APP, capture_output=True,
                       text=True, check=False, env={**os.environ, "VITE_BERKELIUM_API": f"http://127.0.0.1:{api_port}"})
    assert b.returncode == 0, b.stdout + b.stderr
    ui = subprocess.Popen(["npx", "vite", "preview", "--outDir", str(dist), "--host", "127.0.0.1", "--port",
                           str(ui_port), "--strictPort"], cwd=APP, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        _wait_http(f"http://127.0.0.1:{api_port}/health")
        _wait_http(f"http://127.0.0.1:{ui_port}/")
        yield {"api": f"http://127.0.0.1:{api_port}", "ui": f"http://127.0.0.1:{ui_port}"}
    finally:
        for p in (ui, api):
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


def test_prompt_to_mesh_in_studio_viewport(stack):
    from playwright.sync_api import sync_playwright
    health = json.load(urllib.request.urlopen(f"{stack['api']}/health"))
    assert health["components"]["neural"]["status"] == "ok", health
    out_dir = ROOT / "results" / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                          "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        logs = []
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        page.goto(stack["ui"])
        page.get_by_role("tab", name="Scene").click()
        page.wait_for_function("window.__berkeliumViewport && window.__berkeliumViewport.ready", timeout=20000)
        page.get_by_test_id("prompt-input").fill("a red sphere on top of a blue box")
        page.get_by_test_id("generate-btn").click()
        page.get_by_test_id("job-state").filter(has_text="completed").wait_for(timeout=180000)
        page.wait_for_function("window.__berkeliumViewport.faces > 0", timeout=30000)
        vp = page.evaluate("window.__berkeliumViewport")
        card = page.get_by_test_id("artifact-card").inner_text()
        shot = out_dir / "studio_text_to_3d.png"
        page.get_by_test_id("viewport").screenshot(path=str(shot))
        page.get_by_role("button", name="gaussians").click()
        page.wait_for_function("window.__berkeliumViewport.points > 0", timeout=30000)
        vp_g = page.evaluate("window.__berkeliumViewport")
        page.get_by_test_id("viewport").screenshot(path=str(out_dir / "studio_gaussian_centres.png"))
        browser.close()
    rec = json.load(urllib.request.urlopen(f"{stack['api']}/v1/neural/artifacts/{vp['artifactId']}"))
    assert rec["kind"] == "triangle_mesh" and vp["kind"] == "triangle_mesh"
    assert vp["faces"] == rec["validation"]["n_faces"], (vp, rec["validation"])   # same artifact, not a stand-in
    assert "Triangle mesh" in card and "normalized_object" in card
    g = json.load(urllib.request.urlopen(f"{stack['api']}/v1/neural/artifacts/{vp_g['artifactId']}"))
    assert g["kind"] == "gaussian_scene" and vp_g["points"] == g["validation"]["n"]
    import numpy as np
    from PIL import Image
    img = np.asarray(Image.open(shot).convert("RGB"), dtype=np.float32)
    assert img.std() > 5.0, "viewport screenshot is blank"
    (out_dir / "e2e_receipt.json").write_text(json.dumps({"viewport": vp, "gaussian_viewport": vp_g,
                                                          "mesh_artifact": rec["id"], "card": card,
                                                          "console_errors": [x for x in logs if x.startswith("error")]},
                                                         indent=2))
    shutil.copy(shot, out_dir / "latest.png")
