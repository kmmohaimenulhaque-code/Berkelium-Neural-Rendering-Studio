"""Per-prompt adapter routing: apply a concept adapter only when the prompt names its concept.

Why (results/text3d/E-T1_cpu.report.md): the LoRA capsule adapter improves capsule prompts but, applied to every
prompt, degrades prompts without its concept (retention regression). Routing keeps the base model's behaviour on
everything else. One rule, used by the service (adapter="auto") and by the benchmark (E-T2), so what is measured is
what ships.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")


def prompt_words(prompt: str) -> set[str]:
    return set(_WORD.findall(prompt.lower()))


def select_adapter(prompt: str, adapters: dict[str, list[str] | tuple[str, ...]]) -> tuple[str | None, str]:
    """-> (adapter name or None, reason). Deterministic: adapters are checked in name order; the first whose
    concept token appears as a whole word in the prompt wins."""
    words = prompt_words(prompt)
    for name in sorted(adapters):
        hit = [c for c in adapters[name] if c.lower() in words]
        if hit:
            return name, f"prompt names {', '.join(hit)}"
    return None, "no adapter concept in the prompt"
