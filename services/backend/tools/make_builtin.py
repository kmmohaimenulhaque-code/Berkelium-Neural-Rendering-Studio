"""
Writes builtin.json for a release build. Run by the release workflow only.

Reads the key from the environment (a GitHub secret), never from arguments,
so it never shows up in process lists or logs. Prints nothing about the key.

  GROQ_API_KEY=... python tools/make_builtin.py builtin.json

It does not trust a hard-coded model name. Providers retire models, and an
account can have some models switched off, so it sends each candidate one tiny
test message (free tier, a couple of tokens) and ships the first one that
answers. The others that answer ship as backups the app falls back to.
GROQ_MODEL_ID, when set, is tried first.

With no GROQ_API_KEY it writes nothing and exits 0: forks and test builds
simply ship without a built-in key, and the app asks the user for one.
"""

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from providers import scramble  # noqa: E402

GROQ_API = "https://api.groq.com/openai/v1"

# Best first: strongest general chat model, then the others. Only models that
# answer a real test message are shipped.
GROQ_PREFERENCE = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
MAX_BACKUPS = 3
DEFAULT_GROQ_MODEL = GROQ_PREFERENCE[0]

# Speech, safety and other non-chat models the provider lists alongside chat ones.
_NOT_CHAT = ("whisper", "guard", "tts", "orpheus", "playai", "distil", "compound")


class BuildError(Exception):
    pass


def _request(url, key, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "berkelium-release-build"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def candidates(listed, preferred_first=None):
    """Order to try: the explicit choice, then our preference, then anything else
    chat-like the account lists. Never a model the account does not list."""
    listed = [m for m in listed if not any(w in m.lower() for w in _NOT_CHAT)]
    order = ([preferred_first] if preferred_first else []) + GROQ_PREFERENCE + sorted(listed)
    seen, out = set(), []
    for m in order:
        if m and m in listed and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def pick_models(key, preferred_first=None, request=_request, log=print):
    try:
        listed = [m["id"] for m in request(f"{GROQ_API}/models", key).get("data", [])]
    except urllib.error.HTTPError as e:
        raise BuildError(f"Groq refused the model list (HTTP {e.code}). "
                         f"{'The GROQ_API_KEY secret is wrong or revoked.' if e.code in (401, 403) else ''}")
    except (urllib.error.URLError, OSError) as e:
        raise BuildError(f"could not reach Groq: {e}")

    working = []
    for model in candidates(listed, preferred_first):
        try:
            request(f"{GROQ_API}/chat/completions", key, {
                "model": model, "max_tokens": 2,
                "messages": [{"role": "user", "content": "Say OK"}]})
            log(f"make_builtin: {model} answers")
            working.append(model)
            if len(working) > MAX_BACKUPS:
                break
        except urllib.error.HTTPError as e:
            log(f"make_builtin: {model} refused (HTTP {e.code}), skipping")
        except (urllib.error.URLError, OSError) as e:
            log(f"make_builtin: {model} unreachable ({e}), skipping")
    if not working:
        raise BuildError(f"none of the {len(listed)} listed Groq models answered a test message. "
                         "Check the Groq account's model permissions.")
    return working[0], working[1:]


def build(env, request=_request, log=print) -> dict:
    key = env.get("GROQ_API_KEY", "").strip()
    if not key:
        return {}
    model, backups = pick_models(key, env.get("GROQ_MODEL_ID", "").strip() or None, request, log)
    return {"version": 1, "providers": {"groq": {"k": scramble(key), "model": model, "fallbacks": backups}}}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "builtin.json"
    try:
        data = build(os.environ)
    except BuildError as e:
        print(f"::error::make_builtin: {e}")
        sys.exit(1)
    if not data:
        print("make_builtin: no GROQ_API_KEY set, skipping (app will ship without a built-in key)")
        sys.exit(0)
    with open(out, "w") as f:
        json.dump(data, f)
    g = data["providers"]["groq"]
    print(f"make_builtin: wrote {out}: groq model {g['model']}, backups {g['fallbacks'] or 'none'}")
