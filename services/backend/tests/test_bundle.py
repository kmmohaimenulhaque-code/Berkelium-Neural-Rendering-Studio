"""
Tests for running the backend inside the installed desktop app:
the built-in free key, the launcher, and the packaged-app settings.

Nothing here calls a real API. Run from services/backend:  python -m pytest -q
"""

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAKE_KEY = "gsk_test_0123456789abcdefSECRET"


def _fresh_import(monkeypatch, cfg_path, builtin_path=None, log_dir=None):
    monkeypatch.setenv("BERKELIUM_CONFIG", str(cfg_path))
    if builtin_path:
        monkeypatch.setenv("BERKELIUM_BUILTIN", str(builtin_path))
    else:
        monkeypatch.delenv("BERKELIUM_BUILTIN", raising=False)
    if log_dir:
        monkeypatch.setenv("BERKELIUM_LOG_DIR", str(log_dir))
    for mod in ("main", "providers"):
        sys.modules.pop(mod, None)
    import main  # noqa: E402
    from fastapi.testclient import TestClient
    main.limiter.enabled = False
    return main, TestClient(main.app)


def _write_builtin(path, key=FAKE_KEY, model="llama-3.3-70b-versatile"):
    import providers
    path.write_text(json.dumps({"version": 1, "providers": {
        "groq": {"k": providers.scramble(key), "model": model}}}))


@pytest.fixture()
def bundled(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"backend_api_key": "", "allowed_origins": ["null"]}))
    builtin = tmp_path / "builtin.json"
    _write_builtin(builtin)
    main, client = _fresh_import(monkeypatch, cfg, builtin)
    return main, client, cfg


# ------------------------------------------------------------ built-in key

def test_scramble_round_trips_and_hides_the_key():
    import providers
    blob = providers.scramble(FAKE_KEY)
    assert providers.unscramble(blob) == FAKE_KEY
    assert "gsk_" not in blob and "SECRET" not in blob  # scanners must not match it


def test_missing_or_broken_builtin_file_means_none(tmp_path):
    import providers
    assert providers.load_builtin(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert providers.load_builtin(str(bad)) == {}


def test_builtin_skips_unknown_providers_and_corrupt_entries(tmp_path):
    import providers
    f = tmp_path / "b.json"
    f.write_text(json.dumps({"providers": {
        "groq": {"k": "%%%not-base64%%%"},
        "made_up": {"k": providers.scramble("x")},
        "openai": {"k": providers.scramble("sk-good"), "model": "m"},
    }}))
    assert providers.load_builtin(str(f)) == {"openai": {"api_key": "sk-good", "model": "m", "fallbacks": []}}


def test_builtin_makes_groq_ready_and_the_default(bundled):
    _, client, _ = bundled
    r = client.get("/providers")
    body = r.json()
    groq = next(p for p in body["providers"] if p["id"] == "groq")
    assert groq["configured"] is True
    assert groq["builtin"] is True
    assert groq["model"] == "llama-3.3-70b-versatile"
    assert body["default"] == "groq"
    assert groq["values"]["api_key"] == ""  # not even a masked hint of the shipped key
    # The real key must never reach the frontend, masked or not.
    assert FAKE_KEY not in r.text and "SECRET" not in r.text


def test_no_builtin_means_no_default(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({}))
    _, client = _fresh_import(monkeypatch, cfg)
    body = client.get("/providers").json()
    assert body["default"] is None
    assert all(p["builtin"] is False for p in body["providers"])


def test_explicit_default_provider_in_config_wins(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"default_provider": "ollama"}))
    builtin = tmp_path / "builtin.json"
    _write_builtin(builtin)
    _, client = _fresh_import(monkeypatch, cfg, builtin)
    assert client.get("/providers").json()["default"] == "ollama"


def test_users_own_key_wins_and_clearing_falls_back(bundled):
    _, client, cfg = bundled
    client.put("/providers/groq", json={"api_key": "gsk_users_own_key_999"})
    groq = next(p for p in client.get("/providers").json()["providers"] if p["id"] == "groq")
    assert groq["builtin"] is False
    assert groq["values"]["api_key"].endswith("_999"[-4:])

    client.put("/providers/groq", json={"api_key": ""})
    groq = next(p for p in client.get("/providers").json()["providers"] if p["id"] == "groq")
    assert groq["builtin"] is True


def test_builtin_key_is_never_written_to_user_config(bundled):
    _, client, cfg = bundled
    client.put("/providers/groq", json={"model": "some-other-model"})
    on_disk = cfg.read_text()
    assert FAKE_KEY not in on_disk
    assert "_builtin" not in on_disk


def test_chat_uses_builtin_key(bundled, monkeypatch):
    main, client, _ = bundled
    captured = {}
    real = main.prov.call_litellm

    async def spy(p, data, model, messages, timeout, **kw):
        captured["kwargs"] = main.prov._litellm_kwargs(p, data, model)
        return await real(p, data, model, messages, timeout, mock_response="free reply")
    monkeypatch.setattr(main.prov, "call_litellm", spy)

    r = client.post("/chat", json={"message": "hi", "provider": "groq"})
    assert r.status_code == 200, r.text
    assert r.json()["response"] == "free reply"
    assert captured["kwargs"]["api_key"] == FAKE_KEY
    assert captured["kwargs"]["model"] == "groq/llama-3.3-70b-versatile"


def test_builtin_quota_exhausted_has_its_own_code(bundled, monkeypatch):
    main, client, _ = bundled

    async def limited(*a, **kw):
        raise main.prov.ProviderError("upstream_rate_limit", "Groq rate limit hit", 429)
    monkeypatch.setattr(main.prov, "call_litellm", limited)

    r = client.post("/chat", json={"message": "hi", "provider": "groq"})
    assert r.status_code == 429
    assert r.json()["code"] == "builtin_rate_limit"


def test_users_own_key_rate_limit_keeps_normal_code(bundled, monkeypatch):
    main, client, _ = bundled
    client.put("/providers/groq", json={"api_key": "gsk_users_own_key_999"})

    async def limited(*a, **kw):
        raise main.prov.ProviderError("upstream_rate_limit", "Groq rate limit hit", 429)
    monkeypatch.setattr(main.prov, "call_litellm", limited)

    r = client.post("/chat", json={"message": "hi", "provider": "groq"})
    assert r.json()["code"] == "upstream_rate_limit"


# ------------------------------------------------------------ release tool

def _make_builtin():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import make_builtin
    return make_builtin


def fake_groq(listed, answers):
    """A stand-in for Groq: lists `listed`, and only models in `answers` reply."""
    import urllib.error
    calls = []

    def request(url, key, body=None, timeout=30):
        calls.append(body["model"] if body else "LIST")
        if body is None:
            return {"data": [{"id": m} for m in listed]}
        if body["model"] in answers:
            return {"choices": [{"message": {"content": "OK"}}]}
        raise urllib.error.HTTPError(url, 404, "model_not_found", {}, None)
    return request, calls


def test_make_builtin_skips_without_a_key():
    mb = _make_builtin()
    assert mb.build({}) == {}
    assert mb.build({"GROQ_API_KEY": "   "}) == {}


def test_make_builtin_output_loads_back(tmp_path):
    mb = _make_builtin()
    import providers
    req, _ = fake_groq(["llama-3.3-70b-versatile", "openai/gpt-oss-20b"], {"llama-3.3-70b-versatile", "openai/gpt-oss-20b"})
    data = mb.build({"GROQ_API_KEY": FAKE_KEY}, request=req, log=lambda *_: None)
    f = tmp_path / "builtin.json"
    f.write_text(json.dumps(data))
    assert FAKE_KEY not in f.read_text()
    assert providers.load_builtin(str(f)) == {
        "groq": {"api_key": FAKE_KEY, "model": "llama-3.3-70b-versatile", "fallbacks": ["openai/gpt-oss-20b"]}}


def test_make_builtin_skips_models_that_do_not_answer():
    # The exact failure seen in release run 5: the account lists the model but cannot use it.
    mb = _make_builtin()
    req, calls = fake_groq(["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "whisper-large-v3"],
                           {"openai/gpt-oss-120b"})
    model, backups = mb.pick_models(FAKE_KEY, request=req, log=lambda *_: None)
    assert model == "openai/gpt-oss-120b"
    assert backups == []
    assert "whisper-large-v3" not in calls  # speech models are never tried as chat


def test_make_builtin_never_tries_unlisted_models():
    mb = _make_builtin()
    req, calls = fake_groq(["some-new-chat-model"], {"some-new-chat-model"})
    model, _ = mb.pick_models(FAKE_KEY, request=req, log=lambda *_: None)
    assert model == "some-new-chat-model"
    assert calls == ["LIST", "some-new-chat-model"]


def test_make_builtin_explicit_model_goes_first():
    mb = _make_builtin()
    req, _ = fake_groq(["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], {"llama-3.3-70b-versatile", "llama-3.1-8b-instant"})
    model, backups = mb.pick_models(FAKE_KEY, "llama-3.1-8b-instant", request=req, log=lambda *_: None)
    assert (model, backups) == ("llama-3.1-8b-instant", ["llama-3.3-70b-versatile"])


def test_make_builtin_fails_loudly_when_nothing_answers():
    mb = _make_builtin()
    req, _ = fake_groq(["llama-3.3-70b-versatile"], set())
    with pytest.raises(mb.BuildError, match="none of the 1 listed"):
        mb.pick_models(FAKE_KEY, request=req, log=lambda *_: None)


def test_make_builtin_bad_key_says_so():
    import urllib.error
    mb = _make_builtin()

    def request(url, key, body=None, timeout=30):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
    with pytest.raises(mb.BuildError, match="GROQ_API_KEY secret is wrong"):
        mb.pick_models(FAKE_KEY, request=request, log=lambda *_: None)


def test_retired_builtin_model_falls_back(tmp_path, monkeypatch):
    import providers
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    builtin = tmp_path / "builtin.json"
    builtin.write_text(json.dumps({"providers": {"groq": {
        "k": providers.scramble(FAKE_KEY), "model": "old-model", "fallbacks": ["new-model"]}}}))
    main, client = _fresh_import(monkeypatch, cfg, builtin)
    tried = []

    async def fake(p, data, model, messages, timeout, **kw):
        tried.append(model)
        if model == "old-model":
            raise main.prov.ProviderError("model_not_found", "gone", 400)
        return "reply from backup"
    monkeypatch.setattr(main.prov, "call_litellm", fake)

    r = client.post("/chat", json={"message": "hi", "provider": "groq"})
    assert r.status_code == 200, r.text
    assert r.json()["response"] == "reply from backup"
    assert tried == ["old-model", "new-model"]


def test_users_own_key_does_not_use_builtin_fallbacks(tmp_path, monkeypatch):
    import providers
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"providers": {"groq": {"api_key": "gsk_users_own_key_999", "model": "their-model"}}}))
    builtin = tmp_path / "builtin.json"
    builtin.write_text(json.dumps({"providers": {"groq": {
        "k": providers.scramble(FAKE_KEY), "model": "old-model", "fallbacks": ["new-model"]}}}))
    main, client = _fresh_import(monkeypatch, cfg, builtin)
    tried = []

    async def fake(p, data, model, messages, timeout, **kw):
        tried.append(model)
        raise main.prov.ProviderError("model_not_found", "gone", 400)
    monkeypatch.setattr(main.prov, "call_litellm", fake)

    r = client.post("/chat", json={"message": "hi", "provider": "groq"})
    assert r.json()["code"] == "model_not_found"
    assert tried == ["their-model"]  # their key, their model: no silent switching


# ------------------------------------------------------------ launcher

@pytest.mark.parametrize("env,expected", [({}, 8000), ({"BERKELIUM_PORT": "51234"}, 51234)])
def test_port_from_env(env, expected):
    import server
    assert server.port_from_env(env) == expected


@pytest.mark.parametrize("bad", ["0", "70000", "abc", "-5", "80 80"])
def test_port_from_env_rejects_bad_values(bad):
    import server
    with pytest.raises(SystemExit):
        server.port_from_env({"BERKELIUM_PORT": bad})


def test_exit_when_closed_fires_on_eof():
    import server
    fired = []
    server.exit_when_closed(io.BytesIO(b"anything"), lambda: fired.append(True))
    assert fired == [True]


# ------------------------------------------------------------ packaged app

def test_log_dir_env_moves_the_log(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    logs = tmp_path / "app-data" / "logs"
    _fresh_import(monkeypatch, cfg, log_dir=logs)
    assert (logs / "backend.log").exists()


def test_ocr_without_tesseract_is_a_clear_503(bundled, monkeypatch):
    main, client, _ = bundled

    def missing(_contents):
        raise main.pytesseract.TesseractNotFoundError()
    monkeypatch.setattr(main, "_run_image_ocr", missing)

    r = client.post("/ocr", files={"file": ("a.png", b"\x89PNG fake", "image/png")})
    assert r.status_code == 503
    assert r.json()["code"] == "ocr_unavailable"


def test_revoked_builtin_key_has_its_own_code(bundled, monkeypatch):
    main, client, _ = bundled

    async def rejected(*a, **kw):
        raise main.prov.ProviderError("key_rejected", "Groq rejected the API key", 502)
    monkeypatch.setattr(main.prov, "call_litellm", rejected)

    r = client.post("/chat", json={"message": "hi", "provider": "groq"})
    assert r.json()["code"] == "builtin_unavailable"


def test_groq_style_400_bad_key_maps_to_key_rejected(bundled, monkeypatch):
    main, client, _ = bundled
    import litellm
    client.put("/providers/groq", json={"api_key": "gsk_users_own_key_999"})

    async def bad_key(**kw):
        raise litellm.BadRequestError(
            message='GroqException - {"error":{"message":"Invalid API Key","code":"invalid_api_key"}}',
            model="groq/x", llm_provider="groq")
    monkeypatch.setattr(litellm, "acompletion", bad_key)

    r = client.post("/chat", json={"message": "hi", "provider": "groq"})
    assert r.json()["code"] == "key_rejected"
