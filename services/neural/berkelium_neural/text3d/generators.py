"""Text-to-3D generator layer: one protocol, several backends, one routing rule.

  berkelium-textshape  the native 362k-parameter text->SDF model (+ concept adapters routed by routing.py). It reads
                       only its 30-token procedural vocabulary; any other word is <unk>.
  shap-e-text300m      OpenAI Shap-E (MIT) on CPU: open-world prior conditioned on CLIP ViT-L/14 text embeddings.
  zimage-triposr       Z-Image-Turbo -> keyed canonical anchor -> TripoSR (decision D-006, zimage_triposr.py). Needs a
                       GPU (6B diffusion transformer); available() names what is missing on this host. Benchmarked on
                       Kaggle T4 in E-T3.
  auto                 route(): the native model when the prompt has no content word outside its vocabulary,
                       otherwise the first AVAILABLE generator of OPEN_WORLD_RANKING (E-T3 evidence order, D-029);
                       a ranked generator that is skipped because it is unavailable is recorded with its reason.

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
# Open-world generators in E-T3 evidence order (decision D-029, NEURAL_RENDERING_DECISIONS.md). auto sends a prompt
# the native model cannot read to the first of these that is available on this host, and records any it skipped.
# D-029 outcome (E-T3, results/text3d/E-T3_openworld_cpu.json): zimage-triposr (v2) did not beat Shap-E on B1 F@0.05
# (0.603 vs 0.669), B1 Chamfer-L1 (0.055 vs 0.047) or B2 R@1 (0.85 vs 0.95), so Shap-E stays first. It is second:
# a GPU host whose Shap-E weights are missing uses it, and it can always be requested by name.
OPEN_WORLD_RANKING = (SHAP_E, ZIMAGE_TRIPOSR)
ROUTING_RULE = ("native iff every content word is in its vocabulary; otherwise the first available of "
                + " > ".join(OPEN_WORLD_RANKING) + " (E-T3 evidence order, D-029)")


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


# ------------------------------------------------------------------------------------------------ zimage-triposr
class ZImageTripoSR:
    """D-006 image-prior path (text3d/zimage_triposr.py). Imports nothing heavy until generate()."""

    name = ZIMAGE_TRIPOSR

    def __init__(self, **kw):
        self.kw = kw

    def _impl(self):
        from .zimage_triposr import ZImageTripoSRGenerator
        return ZImageTripoSRGenerator(**self.kw)

    def available(self) -> tuple[bool, str]:
        from .zimage_triposr import available
        ok, why = available(self.kw.get("cache"))
        return ok, ("" if ok else "GPU-gated: " + why)

    @property
    def identity(self) -> dict:
        return self._impl().identity

    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict:
        ok, why = self.available()
        if not ok:
            raise GeneratorUnavailable(why)
        return self._impl().generate(prompt, seed, cancel=cancel, progress=progress)


# ---------------------------------------------------------------------------------------------------- route
def content_words(prompt: str) -> list[str]:
    return [w for w in _WORD.findall(prompt.lower()) if w not in STOPWORDS]


def route(prompt: str, vocab) -> dict:
    """The auto rule: native iff every content word is in the native vocabulary. Deterministic, no model calls."""
    known = set(vocab)
    unknown = sorted({w for w in content_words(prompt) if w not in known})
    if unknown:
        return {"generator": OPEN_WORLD_RANKING[0], "ranking": list(OPEN_WORLD_RANKING),
                "reason": f"content words outside the native vocabulary: {', '.join(unknown)}",
                "unknown_words": unknown}
    return {"generator": NATIVE, "reason": "every content word is in the native vocabulary", "unknown_words": []}


def resolve_open_world(available_fn) -> tuple[str | None, list[dict]]:
    """First generator of OPEN_WORLD_RANKING whose ``available_fn(name) -> (ok, why)`` is true, plus the ones skipped
    (with their reasons). (None, skipped) when none is available."""
    skipped = []
    for g in OPEN_WORLD_RANKING:
        ok, why = available_fn(g)
        if ok:
            return g, skipped
        skipped.append({"generator": g, "reason": why})
    return None, skipped


class AutoGenerator:
    """route() then delegate. For an open-world prompt the first available generator in E-T3 evidence order is used
    and every skipped one is recorded in the report; if none is available it fails with all reasons (it never hands
    the prompt to the native model, which cannot read it)."""

    name = AUTO

    def __init__(self, native: NativeTextShape | None = None, open_world=None):
        self.native = native or NativeTextShape()
        self._open = {} if open_world is None else (
            dict(open_world) if isinstance(open_world, dict) else {open_world.name: open_world})

    def _gen(self, name: str):
        if name not in self._open:
            self._open[name] = get_generator(name)
        return self._open[name]

    def available(self) -> tuple[bool, str]:
        return self.native.available()

    @property
    def identity(self) -> dict:
        return {"generator": self.name, "rule": ROUTING_RULE, "ranking": list(OPEN_WORLD_RANKING),
                "stopwords": sorted(STOPWORDS)}

    def decide(self, prompt: str) -> dict:
        return route(prompt, self.native.vocab())

    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict:
        d = self.decide(prompt)
        if d["generator"] == NATIVE:
            gen, skipped = self.native, []
        else:                                   # injected generators (tests, service) stand in for their names
            chosen, skipped = resolve_open_world(lambda n: self._gen(n).available())
            if chosen is None:
                raise GeneratorUnavailable(f"auto routed to {d['generator']}: " +
                                           "; ".join(f"{x['generator']}: {x['reason']}" for x in skipped))
            gen = self._gen(chosen)
        out = gen.generate(prompt, seed, cancel=cancel, progress=progress)
        out["report"]["routing"] = {"requested": AUTO, "selected": gen.name, **d, "skipped_unavailable": skipped}
        out["report"]["generator_identity"] = gen.identity
        return out


def get_generator(name: str, **kw) -> Generator:
    if name == NATIVE:
        return NativeTextShape(**kw)
    if name == SHAP_E:
        from .shap_e_backend import ShapEGenerator
        return ShapEGenerator(**kw)
    if name == ZIMAGE_TRIPOSR:
        return ZImageTripoSR(**kw)
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
                 "identity": AutoGenerator().identity, "routes_to": [NATIVE, *OPEN_WORLD_RANKING],
                 "open_world_now": resolve_open_world(lambda n: (out[n]["available"], out[n]["reason"]))[0]}
    return out
