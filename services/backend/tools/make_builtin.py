"""
Writes builtin.json for a release build. Run by the release workflow only.

Reads the key from the environment (a GitHub secret), never from arguments,
so it never shows up in process lists or logs. Prints nothing about the key.

  GROQ_API_KEY=... GROQ_MODEL_ID=... python tools/make_builtin.py builtin.json

With no GROQ_API_KEY it writes nothing and exits 0: forks and test builds
simply ship without a built-in key, and the app asks the user for one.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from providers import scramble  # noqa: E402

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def build(env) -> dict:
    key = env.get("GROQ_API_KEY", "").strip()
    if not key:
        return {}
    model = env.get("GROQ_MODEL_ID", "").strip() or DEFAULT_GROQ_MODEL
    return {"version": 1, "providers": {"groq": {"k": scramble(key), "model": model}}}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "builtin.json"
    data = build(os.environ)
    if not data:
        print("make_builtin: no GROQ_API_KEY set, skipping (app will ship without a built-in key)")
        sys.exit(0)
    with open(out, "w") as f:
        json.dump(data, f)
    print(f"make_builtin: wrote {out} with providers: {', '.join(data['providers'])}")
