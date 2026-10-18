"""Our own CAD models get their trained prompt, one turn, and greedy decoding.

Everything else keeps the normal chat behaviour. Network calls are faked.
"""

import json

import httpx
import pytest

from test_providers import app_ctx, patch_httpx  # noqa: F401  (pytest fixture)

M32 = "hf.co/Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-GGUF:Q4_K_M"
M8 = "hf.co/Mhaquehaque/berkelium-qwen3-8b-cad-v1-GGUF:latest"


def capture_ollama(monkeypatch, reply="ok"):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": reply}})
    real = httpx.AsyncClient

    def make(*a, **k):
        seen["timeout"] = k.get("timeout")
        return real(*a, **{**k, "transport": httpx.MockTransport(handler)})
    monkeypatch.setattr(httpx, "AsyncClient", make)
    return seen


def test_find_matches_only_our_models():
    import cad_models
    assert cad_models.find(M32).system == cad_models.QWEN3_32B_CADQUERY_PROMPT
    assert cad_models.find(M8).system == cad_models.CAD_01_PROMPT
    assert cad_models.find(M32.upper()) is not None
    for other in ("llama3.2", "qwen3:32b", "", None):
        assert cad_models.find(other) is None


def test_prompts_are_the_trained_ones():
    import cad_models
    # Spot checks against the model repo files, so an accidental edit fails loudly.
    assert cad_models.QWEN3_32B_CADQUERY_PROMPT.startswith("You are a CAD compiler.")
    assert cad_models.QWEN3_32B_CADQUERY_PROMPT.endswith("/no_think")
    assert cad_models.CAD_01_PROMPT.startswith("You are Berkelium's design compiler.")
    assert "structure.cantilever_beam@0.1" in cad_models.CAD_01_PROMPT


@pytest.mark.parametrize("model,prompt_attr", [(M32, "QWEN3_32B_CADQUERY_PROMPT"),
                                                (M8, "CAD_01_PROMPT")])
def test_cad_model_gets_trained_prompt_single_turn(app_ctx, monkeypatch, model, prompt_attr):
    main, client, _ = app_ctx
    seen = capture_ollama(monkeypatch, reply='{"summary":"flange"}')
    r = client.post("/chat", json={
        "message": "Make a flange", "provider": "ollama", "model": model,
        "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}],
    })
    assert r.status_code == 200, r.text
    body = seen["body"]
    assert body["messages"] == [
        {"role": "system", "content": getattr(main.cad_models, prompt_attr)},
        {"role": "user", "content": "Make a flange"},
    ]
    assert body["think"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": 4096, "num_predict": 2048}
    assert body["keep_alive"] == "30m"
    assert body["stream"] is False and body["model"] == model


def test_normal_ollama_model_unchanged(app_ctx, monkeypatch):
    main, client, _ = app_ctx
    seen = capture_ollama(monkeypatch)
    client.post("/chat", json={"message": "hi", "provider": "ollama", "model": "llama3.2"})
    body = seen["body"]
    assert body["messages"][0] == {"role": "system", "content": main.SYSTEM_PROMPT}
    assert not {"think", "options", "keep_alive"} & body.keys()


def test_local_providers_get_long_timeout(app_ctx, monkeypatch):
    main, client, _ = app_ctx
    seen = capture_ollama(monkeypatch)
    client.post("/chat", json={"message": "hi", "provider": "ollama", "model": M32})
    assert seen["timeout"] == main.LOCAL_TIMEOUT_SECONDS == 600.0


def test_cloud_keeps_short_timeout(app_ctx, monkeypatch):
    main, client, _ = app_ctx
    client.put("/providers/openai", json={"api_key": "sk-fake1234567890", "model": "gpt-4o-mini"})
    seen = {}

    async def spy(p, data, model, messages, timeout, **kw):
        seen["timeout"] = timeout
        return "ok"
    monkeypatch.setattr(main.prov, "call_litellm", spy)
    assert client.post("/chat", json={"message": "hi", "provider": "openai"}).status_code == 200
    assert seen["timeout"] == main.FIREWORKS_TIMEOUT_SECONDS


def test_cad_prompt_also_used_on_openai_compatible_server(app_ctx, monkeypatch):
    # e.g. the 32B served by vLLM on an AMD GPU and added as a Custom provider.
    main, client, _ = app_ctx
    client.put("/providers/custom", json={"label": "AMD GPU", "base_url": "http://10.0.0.5:8000/v1",
                                          "api_key": "none-needed-123", "model": M32})
    seen = {}

    async def spy(p, data, model, messages, timeout, **kw):
        seen["messages"] = messages
        return "ok"
    monkeypatch.setattr(main.prov, "call_litellm", spy)
    r = client.post("/chat", json={"message": "Make a flange", "provider": "custom"})
    assert r.status_code == 200, r.text
    assert seen["messages"][0]["content"] == main.cad_models.QWEN3_32B_CADQUERY_PROMPT
    assert len(seen["messages"]) == 2


def test_slow_model_gives_clear_timeout(app_ctx, monkeypatch):
    _, client, _ = app_ctx

    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)}))
    r = client.post("/chat", json={"message": "hi", "provider": "ollama", "model": M32})
    assert r.status_code == 504 and r.json()["code"] == "timeout"
