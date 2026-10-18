"""M8 durable jobs against a REAL server process: SIGKILL mid-job, restart on the same home, the job resumes from its
checkpoints (mesh stage + Gaussian fit), publishes exactly one artifact per stage, and stays retrievable after a
further restart. Missing artifact records are reported, not served."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

NEURAL = Path(__file__).resolve().parents[1]


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _req(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                               method="POST" if body is not None else "GET")
    with urllib.request.urlopen(r, timeout=30) as resp:
        return json.loads(resp.read())


_STARTED: list = []          # every server this module started; killed at teardown even if an assertion failed


@pytest.fixture(autouse=True)
def _no_leaked_servers():
    yield
    while _STARTED:
        p = _STARTED.pop()
        if p.poll() is None:
            p.kill()
            p.wait(10)


class Server:
    def __init__(self, home, models):
        self.port = _port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.env = {**os.environ, "BERKELIUM_NEURAL_HOME": str(home), "BERKELIUM_MODELS": json.dumps(models)}

    def start(self):
        self.p = subprocess.Popen([sys.executable, "-m", "uvicorn", "berkelium_neural.service.api:app", "--port",
                                   str(self.port)], cwd=NEURAL, env=self.env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        _STARTED.append(self.p)
        end = time.time() + 60
        while time.time() < end:
            try:
                return _req(f"{self.url}/health")
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.2)
        raise TimeoutError("server did not start")

    def kill(self):
        self.p.send_signal(signal.SIGKILL)        # a crash: no shutdown hook runs
        self.p.wait(10)

    def stop(self):
        self.p.terminate()
        self.p.wait(30)


def _wait(srv, jid, pred, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        j = _req(f"{srv.url}/v1/neural/jobs/{jid}")
        if pred(j):
            return j
        time.sleep(0.1)
    raise AssertionError(f"timeout; last state {j['state']} {j['stage']}")


def test_sigkill_mid_fit_resumes_from_checkpoints_without_duplicate_artifacts(tiny_run, tmp_path):
    models = {"berkelium-textshape": {"checkpoint": str(tiny_run["base"]["artifact"]), "adapters": {}}}
    srv = Server(tmp_path / "home", models)
    srv.start()
    jid = _req(f"{srv.url}/v1/neural/jobs", {"kind": "text_to_3d", "prompt": "a red sphere", "resolution": 24,
                                             "gaussian_fit_steps": 100})["id"]
    before = _wait(srv, jid, lambda j: any(t["to"] == "checkpointed" for t in j["transitions"]))
    srv.kill()
    assert before["state"] in ("running", "checkpointed") and len(before["artifacts"]) == 1   # mesh published

    srv2 = Server(tmp_path / "home", models)
    srv2.start()
    done = _wait(srv2, jid, lambda j: j["state"] in ("completed", "failed", "cancelled"))
    assert done["state"] == "completed", done.get("error")
    tr = done["transitions"]
    resumed = [t for t in tr if t["to"] == "resumed"]
    assert resumed and resumed[0]["detail"]["after_restart"]
    after = tr[tr.index(resumed[0]):]
    first_ck = next(t for t in after if t["to"] == "checkpointed")
    assert first_ck["detail"]["step"] >= 50, first_ck        # the fit continued from step 25, it did not restart
    kinds = [_req(f"{srv2.url}/v1/neural/artifacts/{a}")["kind"] for a in done["artifacts"]]
    assert sorted(kinds) == ["gaussian_scene", "image", "triangle_mesh"], kinds    # exactly one per stage
    assert done["artifacts"][0] == before["artifacts"][0]                          # the same mesh, reused
    g = _req(f"{srv2.url}/v1/neural/artifacts/{done['artifacts'][1]}")
    assert g["parents"] == [before["artifacts"][0]] and g["config"]["fit"]["steps"] == 100
    assert g["config"]["fit"]["psnr_before"] is not None and g["config"]["fit"]["psnr_after"] is not None
    assert done["missing_artifacts"] == []
    srv2.stop()

    srv3 = Server(tmp_path / "home", models)        # clean restart: still listed and retrievable
    srv3.start()
    again = _req(f"{srv3.url}/v1/neural/jobs/{jid}")
    assert again["state"] == "completed" and again["artifacts"] == done["artifacts"]
    assert jid in [j["id"] for j in _req(f"{srv3.url}/v1/neural/jobs?limit=5")]
    rec = tmp_path / "home" / "artifacts" / "records" / f"{done['artifacts'][2]}.json"
    rec.unlink()                                        # lose a result behind the service's back
    assert _req(f"{srv3.url}/v1/neural/jobs/{jid}")["missing_artifacts"] == [done["artifacts"][2]]
    try:
        _req(f"{srv3.url}/v1/neural/artifacts/{done['artifacts'][2]}")
        raise AssertionError("a missing artifact was served")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    srv3.stop()
