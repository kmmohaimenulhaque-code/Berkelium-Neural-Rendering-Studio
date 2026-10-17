"""The trained output format of the text → CadQuery compiler, and the repair prompt.

A completion is exactly:

    {"summary": "...", "units": "mm", "assumptions": [...], "open_questions": [...]}
    ```python
    import cadquery as cq
    ...
    result = ...
    ```

The header is one line of JSON. The script uses millimetres and only ``cadquery`` and ``math``."""

from __future__ import annotations

import json
import re

FORMAT_VERSION = "cq-v1"
SYSTEM = ("You are a CAD compiler. Turn the request into one line of JSON "
          '{"summary","units":"mm","assumptions":[],"open_questions":[]} followed by one ```python block: a CadQuery '
          "script in millimetres that imports only cadquery and math and ends with `result = ...`. Unstated sizes go "
          "in assumptions; missing key sizes go in open_questions.")
HEADER_KEYS = ("summary", "units", "assumptions", "open_questions")
_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text.strip()}]


def render(header: dict, code: str) -> str:
    h = {"summary": header.get("summary", ""), "units": "mm", "assumptions": list(header.get("assumptions", [])),
         "open_questions": list(header.get("open_questions", []))}
    return json.dumps(h, ensure_ascii=False) + "\n```python\n" + code.strip() + "\n```"


def parse(completion: str) -> tuple[dict | None, str | None, list[str]]:
    """(header, code, problems). Tolerates a stray <think></think> block and a missing closing fence."""
    s = re.sub(r"<think>.*?</think>", "", completion or "", flags=re.S).strip()
    problems: list[str] = []
    header = None
    first = s.split("\n", 1)[0].strip()
    if first.startswith("{"):
        try:
            header = json.loads(first)
        except json.JSONDecodeError as e:
            problems.append(f"header is not JSON: {e.msg}")
    else:
        problems.append("missing JSON header line")
    if isinstance(header, dict):
        miss = [k for k in HEADER_KEYS if k not in header]
        if miss:
            problems.append("header missing " + ", ".join(miss))
        if header.get("units", "mm") != "mm":
            problems.append("units must be mm")
    m = _CODE.search(s)
    code = m.group(1) if m else None
    if code is None and "```python" in s:
        code = s.split("```python", 1)[1]
        problems.append("unterminated code block")
    if code is None:
        problems.append("no ```python block")
    return header if isinstance(header, dict) else None, code, problems


def repair_messages(text: str, completion: str, error: str) -> list[dict]:
    """Ask for a corrected full completion after the sandbox or the core rejected the previous one."""
    return messages(text) + [
        {"role": "assistant", "content": completion},
        {"role": "user", "content": "The script failed:\n" + error.strip()[:1500] +
         "\nReturn the corrected JSON header line and the full corrected ```python script."}]
