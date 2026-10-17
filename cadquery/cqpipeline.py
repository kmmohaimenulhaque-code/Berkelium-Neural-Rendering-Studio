"""One text → CadQuery → sandbox → core attempt, and the repair loop (≤ 2 rounds), shared by eval and Studio."""

from __future__ import annotations

from collections.abc import Callable

import core_check
import cqformat
import sandbox

REPAIR_ROUNDS = 2


def attempt(completion: str, runner) -> dict:
    """Parse, run in the sandbox, validate with the core. ``ok`` = executes + core L3 pass."""
    header, code, problems = cqformat.parse(completion)
    rec = {"completion": completion, "header": header, "format_problems": problems, "ok": False,
           "executed": False, "stage": "format", "error": "; ".join(problems) if code is None else ""}
    if code is None:
        return rec
    res = runner.run(code) if runner is not None else sandbox.run_script(code)
    rec.update(code=code, stage=res.stage, error=res.error, executed=res.ok, volume=res.volume, bbox=res.bbox,
               n_solids=res.n_solids, net_isolation=res.net_isolation, _step=res.step, _glb=res.glb)
    if res.ok and res.step:
        core = core_check.validate(res.step)
        rec["core"] = {k: core[k] for k in ("summary", "l3", "measurements")}
        rec["ok"] = bool(core["l3"])
        if not rec["ok"]:
            rec["stage"] = "core"
            rec["error"] = "; ".join(r["message"] for r in core["results"] if r.get("status") == "fail")[:1500]
    return rec


def error_for_repair(rec: dict) -> str:
    return f"[{rec['stage']}] {rec['error']}".strip()


def compile_text(text: str, chat: Callable[[list[list[dict]]], list[str]], runner=None,
                 rounds: int = REPAIR_ROUNDS) -> dict:
    """First attempt plus up to ``rounds`` repair rounds. Returns the final attempt with the history."""
    completion = chat([cqformat.messages(text)])[0]
    history = []
    for r in range(rounds + 1):
        rec = attempt(completion, runner)
        rec["round"] = r
        history.append({k: v for k, v in rec.items() if not k.startswith("_")})
        if rec["ok"] or r == rounds:
            break
        completion = chat([cqformat.repair_messages(text, completion, error_for_repair(rec))])[0]
    rec["history"] = history
    return rec
