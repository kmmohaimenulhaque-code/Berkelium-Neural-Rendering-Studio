"""Adapter routing (adapter="auto"): an adapter is applied only to prompts that name its concept, in the service
and in the benchmark generator alike."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from berkelium_neural.service.api import NeuralService, create_app
from berkelium_neural.text3d.routing import select_adapter


def test_select_adapter_matches_whole_words_only():
    ads = {"capsule-v1": ["capsule"]}
    assert select_adapter("a green capsule on a box", ads)[0] == "capsule-v1"
    assert select_adapter("A GREEN CAPSULE, on a box", ads)[0] == "capsule-v1"
    assert select_adapter("a red sphere on top of a blue box", ads) == (None, "no adapter concept in the prompt")
    assert select_adapter("encapsulated sphere", ads)[0] is None            # substring is not the concept
    assert select_adapter("a capsule", {})[0] is None


def test_select_adapter_is_deterministic_across_several_adapters():
    ads = {"z-adapter": ["capsule"], "a-adapter": ["capsule", "prism"]}
    assert select_adapter("a capsule", ads)[0] == "a-adapter"
    assert select_adapter("a prism", ads)[0] == "a-adapter"


@pytest.fixture()
def client(tiny_run, tmp_path, monkeypatch):
    monkeypatch.setenv("BERKELIUM_MODELS", json.dumps({"berkelium-textshape": {
        "checkpoint": str(tiny_run["base"]["artifact"]), "adapters": {"capsule-v1": str(tiny_run["lora"]["artifact"])}}}))
    with TestClient(create_app(NeuralService(tmp_path / "home"))) as c:
        yield c


def _mesh_producer(c, prompt):
    jid = c.post("/v1/neural/jobs", json={"kind": "text_to_3d", "prompt": prompt, "adapter": "auto",
                                          "resolution": 24}).json()["id"]
    end = time.time() + 120
    while time.time() < end:
        j = c.get(f"/v1/neural/jobs/{jid}").json()
        if j["state"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.1)
    assert j["state"] == "completed", j.get("error")
    return c.get(f"/v1/neural/artifacts/{j['artifacts'][0]}").json()["producer"]


def test_service_auto_routes_adapter_per_prompt(client):
    caps = client.get("/v1/neural/capabilities").json()
    assert caps["adapter_routing"]["berkelium-textshape"] == {"capsule-v1": ["capsule"]}
    p = _mesh_producer(client, "a green capsule")
    assert p["adapter"] == "capsule-v1" and p["adapter_sha256"]
    assert p["adapter_routing"] == {"requested": "auto", "selected": "capsule-v1", "reason": "prompt names capsule"}
    p = _mesh_producer(client, "a red sphere")
    assert p["adapter"] is None and "adapter_sha256" not in p
    assert p["adapter_routing"]["selected"] is None
