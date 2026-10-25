"""
AI provider registry and routing for Berkelium Studio.

One place that knows about every LLM provider the studio can talk to:
which fields it needs (API key, base URL, AWS creds), how to check that a
key works without spending tokens, and how to send a chat request.

Routing rule agreed with the team:
  * Ollama is called directly over its native HTTP API. That gives us the
    real list of installed models from /api/tags.
  * Everything else (LM Studio, the cloud providers, the custom slot) goes
    through LiteLLM, so we do not maintain fifteen request formats by hand.

Secrets live in config.json under "providers". They never leave the backend.
The frontend only ever receives masked values.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

# Imported lazily inside call_litellm. LiteLLM pulls in a lot at import time
# and the OCR endpoints should not pay that cost on startup.
_litellm = None


def _get_litellm():
    global _litellm
    if _litellm is None:
        import litellm  # noqa: WPS433 (deliberate lazy import)

        litellm.drop_params = True  # ignore params a provider does not support
        litellm.suppress_debug_info = True
        _litellm = litellm
    return _litellm


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    kind: str  # "cloud", "local", or "custom"
    fields: tuple[str, ...]  # config fields the user fills in
    litellm_prefix: Optional[str] = None  # None means not routed via LiteLLM
    default_base_url: Optional[str] = None
    # Free key check. GET this URL; 200 means the credentials work.
    # None means the provider has no free check and we fall back to a
    # 1 token completion, which only runs when the user clicks Test.
    check_url: Optional[str] = None
    check_auth: str = "bearer"  # "bearer", "x-api-key", "query-key", "none"
    # How to pull model ids out of the check_url response, when it lists models.
    models_shape: Optional[str] = None  # "openai", "anthropic", "gemini", "cohere", "ollama"
    default_model: str = ""
    docs_url: str = ""
    extra_headers: dict = field(default_factory=dict)


# Order here is the order the Settings window shows them.
# default_model is only a starting suggestion. Users can type any model id
# their account has access to, and the Test button lists the real ones.
PROVIDERS: dict[str, Provider] = {p.id: p for p in [
    # Local runners first. They are free and the easiest way to demo.
    Provider("ollama", "Ollama", "local", ("base_url",),
             default_base_url="http://localhost:11434",
             check_url="{base_url}/api/tags", check_auth="none", models_shape="ollama",
             docs_url="https://ollama.com"),
    Provider("lm_studio", "LM Studio", "local", ("base_url",), "lm_studio",
             default_base_url="http://localhost:1234/v1",
             check_url="{base_url}/models", check_auth="none", models_shape="openai",
             docs_url="https://lmstudio.ai"),

    # Cloud providers. Each one needs the user's own key.
    Provider("openai", "OpenAI", "cloud", ("api_key",), "openai",
             check_url="https://api.openai.com/v1/models", models_shape="openai",
             default_model="gpt-4o-mini", docs_url="https://platform.openai.com/api-keys"),
    Provider("anthropic", "Anthropic", "cloud", ("api_key",), "anthropic",
             check_url="https://api.anthropic.com/v1/models", check_auth="x-api-key",
             models_shape="anthropic", default_model="claude-sonnet-4-5",
             docs_url="https://console.anthropic.com/settings/keys",
             extra_headers={"anthropic-version": "2023-06-01"}),
    Provider("gemini", "Google Gemini", "cloud", ("api_key",), "gemini",
             check_url="https://generativelanguage.googleapis.com/v1beta/models",
             check_auth="query-key", models_shape="gemini", default_model="gemini-2.5-flash",
             docs_url="https://aistudio.google.com/apikey"),
    Provider("fireworks", "Fireworks AI", "cloud", ("api_key",), "fireworks_ai",
             check_url="https://api.fireworks.ai/inference/v1/models", models_shape="openai",
             docs_url="https://fireworks.ai/account/api-keys"),
    Provider("together", "Together AI", "cloud", ("api_key",), "together_ai",
             check_url="https://api.together.xyz/v1/models", models_shape="openai",
             docs_url="https://api.together.ai/settings/api-keys"),
    Provider("mistral", "Mistral", "cloud", ("api_key",), "mistral",
             check_url="https://api.mistral.ai/v1/models", models_shape="openai",
             default_model="mistral-small-latest", docs_url="https://console.mistral.ai/api-keys"),
    Provider("groq", "Groq", "cloud", ("api_key",), "groq",
             check_url="https://api.groq.com/openai/v1/models", models_shape="openai",
             docs_url="https://console.groq.com/keys"),
    Provider("openrouter", "OpenRouter", "cloud", ("api_key",), "openrouter",
             # /models is public, so it would not prove the key works. /key does.
             check_url="https://openrouter.ai/api/v1/key",
             docs_url="https://openrouter.ai/keys"),
    Provider("cohere", "Cohere", "cloud", ("api_key",), "cohere_chat",
             check_url="https://api.cohere.com/v1/models", models_shape="cohere",
             docs_url="https://dashboard.cohere.com/api-keys"),
    Provider("deepseek", "DeepSeek", "cloud", ("api_key",), "deepseek",
             check_url="https://api.deepseek.com/models", models_shape="openai",
             default_model="deepseek-chat", docs_url="https://platform.deepseek.com/api_keys"),
    Provider("perplexity", "Perplexity", "cloud", ("api_key",), "perplexity",
             default_model="sonar", docs_url="https://www.perplexity.ai/settings/api"),
    Provider("xai", "xAI (Grok)", "cloud", ("api_key",), "xai",
             check_url="https://api.x.ai/v1/models", models_shape="openai",
             docs_url="https://console.x.ai"),
    Provider("nvidia_nim", "NVIDIA NIM", "cloud", ("api_key",), "nvidia_nim",
             default_base_url="https://integrate.api.nvidia.com/v1",
             check_url="{base_url}/models", models_shape="openai",
             default_model="meta/llama-3.1-8b-instruct",
             docs_url="https://build.nvidia.com/settings/api-keys"),
    Provider("huggingface", "Hugging Face", "cloud", ("api_key",), "huggingface",
             check_url="https://huggingface.co/api/whoami-v2",
             docs_url="https://huggingface.co/settings/tokens"),
    Provider("replicate", "Replicate", "cloud", ("api_key",), "replicate",
             check_url="https://api.replicate.com/v1/account",
             docs_url="https://replicate.com/account/api-tokens"),
    Provider("bedrock", "AWS Bedrock", "cloud",
             ("aws_access_key_id", "aws_secret_access_key", "aws_region_name"), "bedrock",
             docs_url="https://console.aws.amazon.com/bedrock"),

    # Any OpenAI-compatible endpoint: a private gateway, vLLM, a new provider.
    Provider("custom", "Custom (OpenAI-compatible)", "custom",
             ("label", "base_url", "api_key"), "openai",
             check_url="{base_url}/models", models_shape="openai"),
]}

SECRET_FIELDS = {"api_key", "aws_access_key_id", "aws_secret_access_key"}
ALL_FIELDS = SECRET_FIELDS | {"base_url", "aws_region_name", "label", "model"}
MAX_FIELD_LEN = 512


class ProviderError(Exception):
    """Raised with a short machine code the frontend maps to a message."""

    def __init__(self, code: str, message: str, status: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# ---------------------------------------------------------------------------
# Built-in keys shipped inside the installer
# ---------------------------------------------------------------------------
# The release build writes builtin.json next to this file so chat works on a
# fresh install with no setup. It lives outside config.json on purpose:
#   * The user's own key always wins. Clearing it falls back to the built-in.
#   * A new release rotates the built-in key without touching user settings.
#   * The key is never copied into the user's config file on disk.
# The key is XOR scrambled. That only stops automated key scanners from
# matching it and revoking it. It does NOT stop a person who wants the key,
# so only keys from accounts that cannot be billed may ever go in here.

BUILTIN_FILE = "builtin.json"
_SCRAMBLE = b"berkelium-builtin-v1"


def scramble(text: str) -> str:
    import base64
    raw = text.encode("utf-8")
    out = bytes(b ^ _SCRAMBLE[i % len(_SCRAMBLE)] for i, b in enumerate(raw))
    return base64.b64encode(out).decode("ascii")


def unscramble(blob: str) -> str:
    import base64
    raw = base64.b64decode(blob.encode("ascii"))
    return bytes(b ^ _SCRAMBLE[i % len(_SCRAMBLE)] for i, b in enumerate(raw)).decode("utf-8")


def load_builtin(path: str) -> dict:
    """{provider_id: {"api_key": ..., "model": ...}}. A missing or broken file
    means no built-in keys. That is the normal case in development."""
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    out = {}
    for pid, entry in (raw.get("providers") or {}).items():
        if pid not in PROVIDERS or not isinstance(entry, dict):
            continue
        try:
            key = unscramble(entry.get("k", ""))
        except Exception:  # noqa: BLE001 (a corrupt entry is skipped, not fatal)
            continue
        if key:
            backups = entry.get("fallbacks")
            out[pid] = {"api_key": key, "model": str(entry.get("model", "")),
                        "fallbacks": [str(m) for m in backups] if isinstance(backups, list) else []}
    return out


# ---------------------------------------------------------------------------
# Config storage
# ---------------------------------------------------------------------------

class ProviderStore:
    """Reads and writes the "providers" section of config.json.

    Writes are atomic (temp file then rename) so a crash mid-save can never
    leave config.json half written. Other top-level keys are preserved.
    """

    def __init__(self, config_path: str, legacy_config: dict, builtin: Optional[dict] = None):
        self.path = config_path
        self._lock = threading.Lock()
        self._legacy = legacy_config
        self.builtin = builtin or {}

    def _read(self) -> dict:
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def get(self, provider_id: str) -> dict:
        data = self._read().get("providers", {}).get(provider_id, {})
        data = dict(data) if isinstance(data, dict) else {}
        # Backward compat: the original single-provider config kept the
        # Fireworks key and model at the top level. Keep honoring them.
        if provider_id == "fireworks":
            data.setdefault("api_key", self._legacy.get("fireworks_api_key", ""))
            data.setdefault("model", self._legacy.get("model_id", ""))
        # Fall back to the shipped key only when the user has none of their
        # own. "_builtin" is internal: public_view turns it into a flag and it
        # is never written to disk, because update() starts from the raw file.
        shipped = self.builtin.get(provider_id)
        if shipped:
            # The shipped model is a sensible default even for a user who
            # pasted their own key but never picked a model.
            if not data.get("model") and shipped["model"]:
                data["model"] = shipped["model"]
            if not data.get("api_key"):
                data["api_key"] = shipped["api_key"]
                data["_builtin"] = True
                data["_fallbacks"] = list(shipped.get("fallbacks", []))
        return data

    def update(self, provider_id: str, changes: dict) -> dict:
        with self._lock:
            full = self._read()
            providers = full.setdefault("providers", {})
            current = dict(providers.get(provider_id, {}))
            for k, v in changes.items():
                if v is None:
                    continue  # field not sent, leave it alone
                if v == "":
                    current.pop(k, None)  # explicit clear
                else:
                    current[k] = v
            providers[provider_id] = current
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(full, f, indent=2)
            os.replace(tmp, self.path)
            return current


def mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "••••"
    return f"{value[:3]}••••{value[-4:]}"


def is_configured(p: Provider, data: dict) -> bool:
    if p.kind == "local":
        return True  # defaults to localhost; reachability is a separate check
    needed = [f for f in p.fields if f != "label"]
    return all(data.get(f) for f in needed)


def public_view(p: Provider, data: dict) -> dict:
    """What the frontend is allowed to see. Secrets come back masked."""
    values = {}
    for f in p.fields:
        raw = data.get(f, "")
        # The shipped key is not the user's, so Settings shows nothing for
        # it (not even masked) and offers no "Remove" that would do nothing.
        if f in SECRET_FIELDS and data.get("_builtin"):
            raw = ""
        if f == "base_url" and not raw:
            raw = p.default_base_url or ""
        values[f] = mask(raw) if f in SECRET_FIELDS else raw
    return {
        "id": p.id,
        "label": (data.get("label") or p.label) if p.kind == "custom" else p.label,
        "kind": p.kind,
        "fields": list(p.fields),
        "values": values,
        "model": data.get("model") or p.default_model,
        "configured": is_configured(p, data),
        "can_list_models": bool(p.models_shape),
        "docs_url": p.docs_url,
        # True while the shipped free key is in use, so Settings can say so.
        "builtin": bool(data.get("_builtin")),
    }


# ---------------------------------------------------------------------------
# Key check and model listing (free, no tokens spent)
# ---------------------------------------------------------------------------

def _base_url(p: Provider, data: dict) -> str:
    return (data.get("base_url") or p.default_base_url or "").rstrip("/")


def _parse_models(shape: str, body: Any) -> list[str]:
    try:
        if shape == "openai":
            return sorted(m["id"] for m in body.get("data", []))
        if shape == "anthropic":
            return [m["id"] for m in body.get("data", [])]
        if shape == "gemini":
            out = []
            for m in body.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    out.append(m["name"].removeprefix("models/"))
            return out
        if shape == "cohere":
            return [m["name"] for m in body.get("models", []) if "chat" in m.get("endpoints", ["chat"])]
        if shape == "ollama":
            return [m["name"] for m in body.get("models", [])]
    except (KeyError, TypeError, AttributeError):
        pass
    return []


async def check_provider(p: Provider, data: dict, timeout: float = 8.0) -> dict:
    """Prove the credentials work and, where possible, list real models.

    Uses a free GET endpoint for every provider that has one. Returns
    {"ok": bool, "models": [...], "detail": str}.
    """
    if not p.check_url:
        return {"ok": None, "models": [], "detail": "no_free_check"}
    if not is_configured(p, data):
        return {"ok": False, "models": [], "detail": "not_configured"}

    url = p.check_url.format(base_url=_base_url(p, data))
    headers = dict(p.extra_headers)
    params = {}
    key = data.get("api_key", "")
    if p.check_auth == "bearer" and key:
        headers["Authorization"] = f"Bearer {key}"
    elif p.check_auth == "x-api-key":
        headers["x-api-key"] = key
    elif p.check_auth == "query-key":
        params["key"] = key

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers=headers, params=params)
    except httpx.ConnectError:
        detail = "not_running" if p.kind == "local" else "unreachable"
        return {"ok": False, "models": [], "detail": detail}
    except httpx.RequestError:
        return {"ok": False, "models": [], "detail": "unreachable"}

    # Gemini and xAI answer a bad key with 400, not 401. This is a bodiless
    # GET, so a 400 can only mean the credentials were refused.
    if r.status_code in (400, 401, 403):
        return {"ok": False, "models": [], "detail": "key_rejected"}
    if r.status_code != 200:
        return {"ok": False, "models": [], "detail": f"status_{r.status_code}"}

    models = []
    if p.models_shape:
        try:
            models = _parse_models(p.models_shape, r.json())
        except ValueError:
            models = []
    return {"ok": True, "models": models, "detail": "ok"}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

async def call_ollama(p: Provider, data: dict, model: str, messages: list[dict],
                      timeout: float, extra: Optional[dict] = None) -> str:
    url = f"{_base_url(p, data)}/api/chat"
    body = {"model": model, "messages": messages, "stream": False}
    # Per-model settings (think, options, keep_alive). The core fields above win.
    body = {**(extra or {}), **body}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=body)
    except httpx.ConnectError:
        raise ProviderError("not_running", "Ollama is not running", 503)
    except httpx.TimeoutException:
        raise ProviderError("timeout", "The model took too long to answer", 504)
    except httpx.RequestError as e:
        raise ProviderError("unreachable", f"Ollama request failed: {e}", 502)
    if r.status_code == 404:
        raise ProviderError("model_not_found", f"Ollama has no model '{model}'", 400)
    if r.status_code != 200:
        raise ProviderError("upstream_error", f"Ollama returned {r.status_code}", 502)
    try:
        return r.json()["message"]["content"]
    except (ValueError, KeyError, TypeError):
        raise ProviderError("bad_response", "Ollama sent an unexpected response", 502)


async def call_ollama_raw(p: Provider, data: dict, model: str, prompt: str, options: dict,
                          keep_alive: str, timeout: float) -> str:
    """Send a fully formatted prompt to Ollama with no template applied.

    Streams the answer and joins it. A long answer from a slow machine then keeps
    the connection busy instead of sitting idle until something cuts it off.
    """
    url = f"{_base_url(p, data)}/api/generate"
    body = {"model": model, "prompt": prompt, "raw": True, "stream": True,
            "options": options, "keep_alive": keep_alive}
    parts: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=body) as r:
                if r.status_code == 404:
                    raise ProviderError("model_not_found", f"Ollama has no model '{model}'", 400)
                if r.status_code != 200:
                    raise ProviderError("upstream_error", f"Ollama returned {r.status_code}", 502)
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        raise ProviderError("bad_response", "Ollama sent an unexpected response", 502)
                    if chunk.get("error"):
                        raise ProviderError("upstream_error", f"Ollama error: {chunk['error']}", 502)
                    parts.append(chunk.get("response", ""))
                    if chunk.get("done"):
                        break
    except httpx.ConnectError:
        raise ProviderError("not_running", "Ollama is not running", 503)
    except httpx.TimeoutException:
        raise ProviderError("timeout", "The model took too long to answer", 504)
    except httpx.RequestError as e:
        raise ProviderError("unreachable", f"Ollama request failed: {e}", 502)
    return "".join(parts)


def _litellm_kwargs(p: Provider, data: dict, model: str) -> dict:
    kwargs: dict[str, Any] = {"model": f"{p.litellm_prefix}/{model}"}
    if data.get("api_key"):
        kwargs["api_key"] = data["api_key"]
    if p.kind in ("local", "custom") or p.id == "nvidia_nim":
        kwargs["api_base"] = _base_url(p, data)
        if p.id == "lm_studio" and "api_key" not in kwargs:
            kwargs["api_key"] = "lm-studio"  # LM Studio ignores it, LiteLLM wants one
    if p.id == "bedrock":
        for f in ("aws_access_key_id", "aws_secret_access_key", "aws_region_name"):
            kwargs[f] = data.get(f)
    return kwargs


async def call_litellm(p: Provider, data: dict, model: str, messages: list[dict],
                       timeout: float, max_tokens: int = 1024,
                       mock_response: Optional[str] = None) -> str:
    litellm = _get_litellm()
    kwargs = _litellm_kwargs(p, data, model)
    if mock_response is not None:
        kwargs["mock_response"] = mock_response
    try:
        resp = await litellm.acompletion(
            messages=messages, max_tokens=max_tokens, temperature=0.7,
            timeout=timeout, num_retries=2, **kwargs,
        )
    except litellm.AuthenticationError:
        raise ProviderError("key_rejected", f"{p.label} rejected the API key", 502)
    except litellm.NotFoundError:
        raise ProviderError("model_not_found", f"{p.label} has no model '{model}'", 400)
    except litellm.RateLimitError:
        raise ProviderError("upstream_rate_limit", f"{p.label} rate limit hit", 429)
    except litellm.APIConnectionError:
        code = "not_running" if p.kind == "local" else "unreachable"
        raise ProviderError(code, f"Could not reach {p.label}", 503)
    except litellm.BadRequestError as e:
        # Groq (and some others) answer a bad key with 400, not 401. Without
        # this the user would see a raw provider message instead of "fix
        # your key in Settings".
        if "invalid_api_key" in str(e) or "Invalid API Key" in str(e):
            raise ProviderError("key_rejected", f"{p.label} rejected the API key", 502)
        raise ProviderError("bad_request", f"{p.label} rejected the request: {e}", 400)
    except Exception as e:  # noqa: BLE001 (map anything else to a clean 502)
        raise ProviderError("upstream_error", f"{p.label} failed: {e}", 502)
    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError):
        raise ProviderError("bad_response", f"{p.label} sent an unexpected response", 502)


async def chat(p: Provider, data: dict, model: str, messages: list[dict],
               timeout: float = 60.0, ollama_extra: Optional[dict] = None,
               ollama_raw: Optional[dict] = None) -> str:
    if not is_configured(p, data):
        raise ProviderError("not_configured", f"{p.label} is not set up yet", 400)
    if not model:
        raise ProviderError("no_model", f"Pick a model for {p.label}", 400)
    if p.id == "ollama":
        if ollama_raw:
            return await call_ollama_raw(p, data, model, timeout=timeout, **ollama_raw)
        return await call_ollama(p, data, model, messages, timeout, ollama_extra)
    # Providers retire models. For the shipped key, a vanished model falls back
    # to the backups the release build tested, so users never see the error.
    tries = [model]
    if data.get("_builtin"):
        tries += [m for m in data.get("_fallbacks", []) if m != model]
    try:
        for i, m in enumerate(tries):
            try:
                return await call_litellm(p, data, m, messages, timeout)
            except ProviderError as e:
                if e.code != "model_not_found" or i == len(tries) - 1:
                    raise
    except ProviderError as e:
        # The shipped key is one free quota shared by every install. When it
        # runs dry the fix is the user's own key, not "wait a minute", so the
        # frontend needs a different code to say that.
        if data.get("_builtin") and e.code == "upstream_rate_limit":
            raise ProviderError("builtin_rate_limit",
                                f"The free {p.label} quota is used up for now", 429) from e
        # A revoked or blocked shipped key: the user did nothing wrong, so
        # do not tell them "your key" was rejected.
        if data.get("_builtin") and e.code == "key_rejected":
            raise ProviderError("builtin_unavailable",
                                f"The free built-in {p.label} key no longer works", 502) from e
        raise
