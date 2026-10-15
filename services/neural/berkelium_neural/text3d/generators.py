"""Text-to-3D generator layer: one protocol, several backends, one routing rule.

  berkelium-textshape  the native 362k-parameter text->SDF model (+ concept adapters routed by routing.py). It reads
                       only its 30-token procedural vocabulary; any other word is <unk>.
  shap-e-text300m      OpenAI Shap-E (MIT) on CPU: open-world prior conditioned on CLIP ViT-L/14 text embeddings.
  zimage-triposr       Z-Image-Turbo -> TripoSR (decision D-006). Registered but GPU-gated: available() is False with
                       the M5 blocker (results/m5/E-A1_preflight.json) as the reason, and it never emits output.
  auto                 route(): the native model when the prompt has no content word outside its vocabulary,
                       otherwise Shap-E. Measured by E-T3 (results/text3d/E-T3_openworld_cpu.json).

Every backend returns {vertices, faces, colors, report} in normalized_object units (+Y up, centred, max extent 1.6),
and exposes an ``identity`` dict (model, commit, weight sha256, licence, sampler) that goes into artifact provenance.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

REPO = Path(__file__).resolve().parents[4]
NATIVE, SHAP_E, ZIMAGE_TRIPOSR, AUTO = "berkelium-textshape", "shap-e-text300m", "zimage-triposr", "auto"
# Function words that carry no shape content. Everything else outside the native vocabulary is a content word
# the native model cannot read (it becomes <unk>).
STOPWORDS = frozenset({"a", "an", "the", "and", "of", "in", "is", "it", "its", "at", "by", "for", "from", "as"})
_WORD = re.compile(r"[a-z]+")      # the native tokenizer's own word rule (model.tokenize)


class GeneratorUnavailable(RuntimeError):
    code = "MODEL_UNAVAILABLE"


@runtime_checkable
class Generator(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...

    @property
    def identity(self) -> dict: ...

    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict: ...


# ------------------------------------------------------------------------------------------------- native
class NativeTextShape:
    """The existing model, unchanged: text -> SDF -> marching cubes. ``adapter`` = None | name | "auto"."""

    name = NATIVE

    def __init__(self, checkpoint: str | Path | None = None, adapters: dict | None = None, adapter: str | None = "auto",
                 resolution: int = 48):
        self.checkpoint = Path(checkpoint or REPO / "exports/textshape-base-v1/model.pt")
        self.adapters = {k: Path(v) for k, v in (adapters if adapters is not None else
                                                 {"capsule-v1": REPO / "exports/textshape-lora-capsule-v1/adapter.pt"}
                                                 ).items()}
        self.adapter, self.resolution = adapter, resolution

    def available(self) -> tuple[bool, str]:
        if not self.checkpoint.exists():
            return False, f"native checkpoint missing: {self.checkpoint}"
        return True, ""

    @property
    def identity(self) -> dict:
        return {"generator": self.name, "model": "BerkeliumTextShape (text->SDF->marching cubes)",
                "checkpoint": str(self.checkpoint.relative_to(REPO)) if self.checkpoint.is_relative_to(REPO)
                else str(self.checkpoint), "license": "MIT (Berkelium-owned weights and data)",
                "resolution": self.resolution, "adapter": self.adapter, "device": "cpu"}

    def vocab(self) -> list[str]:
        return list(_native_model(str(self.checkpoint), None).vocab)

    def _adapter_for(self, prompt: str) -> tuple[str | None, dict | None]:
        if self.adapter != "auto":
            return self.adapter, None
        from .routing import select_adapter
        concepts = {n: list(_adapter_concepts(str(p))) for n, p in self.adapters.items() if p.exists()}
        chosen, reason = select_adapter(prompt, concepts)
        return chosen, {"requested": "auto", "selected": chosen, "reason": reason}

    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict:
        from .generate import generate_mesh
        adapter, routing = self._adapter_for(prompt)
        if progress:
            progress(0.0, "loading")
        model = _native_model(str(self.checkpoint), str(self.adapters[adapter]) if adapter else None)
        if progress:
            progress(0.1, "sampling")
        out = generate_mesh(model, prompt, self.resolution, cancel=cancel)    # deterministic; seed recorded only
        rep = out["report"]
        rep.update({"generator": self.name, "seed": seed, "adapter": adapter, **model.identity,
                    **({"adapter_routing": routing} if routing else {})})
        if progress:
            progress(0.95, "validating")
        return out


@lru_cache(maxsize=4)
def _native_model(ckpt: str, adapter: str | None):
    from .train import load_model
    return load_model(Path(ckpt), Path(adapter) if adapter else None)


@lru_cache(maxsize=8)
def _adapter_concepts(path: str) -> tuple[str, ...]:
    import torch
    return tuple(torch.load(path, map_location="cpu", weights_only=False)["adapter_config"]["new_tokens"])


# ------------------------------------------------------------------------------------------------ GPU-gated
class ZImageTripoSR:
    """D-006 image-prior path. Not runnable here: see results/m5/E-A1_preflight.json. Never emits output."""

    name = ZIMAGE_TRIPOSR
    PREFLIGHT = REPO / "results/m5/E-A1_preflight.json"

    def available(self) -> tuple[bool, str]:
        import json
        try:
            pf = json.loads(self.PREFLIGHT.read_text())
            blockers = pf.get("preflight", {}).get("blockers") or [pf.get("status", "unknown")]
        except (OSError, ValueError):
            blockers = ["no M5 preflight record"]
        return False, "GPU-gated (M5 blocked_environment): " + "; ".join(blockers)

    @property
    def identity(self) -> dict:
        return {"generator": self.name, "model": "Z-Image-Turbo (f332072a) -> TripoSR", "status": "gpu_gated",
                "license": "Apache-2.0 (Z-Image-Turbo model card) + MIT (TripoSR)"}

    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict:
        raise GeneratorUnavailable(self.available()[1])


# ---------------------------------------------------------------------------------------------------- route
def content_words(prompt: str) -> list[str]:
    return [w for w in _WORD.findall(prompt.lower()) if w not in STOPWORDS]


def route(prompt: str, vocab) -> dict:
    """The auto rule: native iff every content word is in the native vocabulary. Deterministic, no model calls."""
    known = set(vocab)
    unknown = sorted({w for w in content_words(prompt) if w not in known})
    if unknown:
        return {"generator": SHAP_E, "reason": f"content words outside the native vocabulary: {', '.join(unknown)}",
                "unknown_words": unknown}
    return {"generator": NATIVE, "reason": "every content word is in the native vocabulary", "unknown_words": []}


class AutoGenerator:
    """route() then delegate. If Shap-E is chosen but unavailable, fails with the reason (never silently falls back to
    a model that cannot read the prompt)."""

    name = AUTO

    def __init__(self, native: NativeTextShape | None = None, open_world=None):
        self.native = native or NativeTextShape()
        self.open_world = open_world

    def _open_world(self):
        if self.open_world is None:
            from .shap_e_backend import ShapEGenerator
            self.open_world = ShapEGenerator()
        return self.open_world

    def available(self) -> tuple[bool, str]:
        return self.native.available()

    @property
    def identity(self) -> dict:
        return {"generator": self.name, "rule": "native iff no content word outside its vocabulary, else shap-e",
                "stopwords": sorted(STOPWORDS)}

    def decide(self, prompt: str) -> dict:
        return route(prompt, self.native.vocab())

    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict:
        d = self.decide(prompt)
        gen = self.native if d["generator"] == NATIVE else self._open_world()
        ok, why = gen.available()
        if not ok:
            raise GeneratorUnavailable(f"auto routed to {gen.name}: {why}")
        out = gen.generate(prompt, seed, cancel=cancel, progress=progress)
        out["report"]["routing"] = {"requested": AUTO, "selected": gen.name, **d}
        out["report"]["generator_identity"] = gen.identity
        return out


def get_generator(name: str, **kw) -> Generator:
    if name == NATIVE:
        return NativeTextShape(**kw)
    if name == SHAP_E:
        from .shap_e_backend import ShapEGenerator
        return ShapEGenerator(**kw)
    if name == ZIMAGE_TRIPOSR:
        return ZImageTripoSR()
    if name == AUTO:
        return AutoGenerator(**kw)
    raise GeneratorUnavailable(f"unknown generator {name!r}")


def generator_status() -> dict:
    """name -> {available, reason, identity} for /capabilities and the Studio selector. Loads no weights."""
    out = {}
    for name in (NATIVE, SHAP_E, ZIMAGE_TRIPOSR):
        g = get_generator(name)
        ok, why = g.available()
        out[name] = {"available": ok, "reason": why, "identity": g.identity}
    out[AUTO] = {"available": out[NATIVE]["available"], "reason": out[NATIVE]["reason"],
                 "identity": AutoGenerator().identity, "routes_to": [NATIVE, SHAP_E]}
    return out
