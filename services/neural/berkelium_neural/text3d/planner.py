"""Deterministic two-object scene planner (no LLM): parse a relation, generate each object separately, place them.

Grammar (lower-cased, whitespace-normalised prompt; A and B are free-text noun phrases that must each start with
a/an/the/one, so "a black and white cow" stays one object; each is generated on its own):
  on_top_of   "A on top of B" | "A stacked on (top of) B" | "B with A on top" | "A on B"
  next_to     "A next to B" | "A beside B" | "A to the left of B" | "A left of B" | "A and B"
              "A to the right of B" | "A right of B"  (same relation, B placed on the left)
Placement follows the conventions of the procedural ground truth (shapes.placements): each part is rescaled so its
largest side is PART_EXTENT x size factor ("small" 0.7, "large" 1.3, as shapes.SIZES), A sits on B touching at the
bottom of A (x/z centres aligned) or A is left of B (-X) with a GAP between bounding boxes (y/z centres aligned); the
assembled scene is then centred and scaled so its largest side is EXTENT (1.6). Every transform is explicit
(4x4, part -> scene) and each part is validated on its own. Parts are never merged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

PART_EXTENT = 1.0     # pre-normalisation size of a default part (a default sphere in shapes.py has diameter 1.0)
GAP = 0.2             # = shapes.GAP
EXTENT = 1.6          # = shapes.EXTENT
SIZE_FACTOR = {"small": 0.7, "large": 1.3}   # = shapes.SIZES

_RULES = (   # (pattern, relation, swap): swap=True means the first group is the bottom/left object
    (r"^(?P<a>.+?) on top of (?P<b>.+)$", "on_top_of", False),
    (r"^(?P<a>.+?) stacked on (?:top of )?(?P<b>.+)$", "on_top_of", False),
    (r"^(?P<b>.+?) with (?P<a>.+?) on top$", "on_top_of", False),
    (r"^(?P<a>.+?) next to (?P<b>.+)$", "next_to", False),
    (r"^(?P<a>.+?) beside (?P<b>.+)$", "next_to", False),
    (r"^(?P<a>.+?) (?:to the )?left of (?P<b>.+)$", "next_to", False),
    (r"^(?P<a>.+?) (?:to the )?right of (?P<b>.+)$", "next_to", True),
    (r"^(?P<a>.+?) on (?P<b>.+)$", "on_top_of", False),
    (r"^(?P<a>.+?) and (?P<b>.+)$", "next_to", False),
)
_RELATION_WORDS = re.compile(r"\b(on top of|stacked on|on top|next to|beside|left of|right of|on|and)\b")
_DETERMINER = re.compile(r"^(a|an|the|one)\b")   # both objects must be noun phrases ("a black and white cow" is one)


@dataclass(frozen=True)
class Plan:
    relation: str            # on_top_of | next_to
    a: str                   # top / left object
    b: str                   # bottom / right object
    rule: str

    def as_dict(self) -> dict:
        return {"relation": self.relation, "a": self.a, "b": self.b, "rule": self.rule,
                "roles": {"a": "top" if self.relation == "on_top_of" else "left",
                          "b": "bottom" if self.relation == "on_top_of" else "right"}}


def parse(prompt: str) -> tuple[Plan | None, str]:
    """-> (plan, reason). None when the prompt is not exactly two objects in a supported relation."""
    p = " ".join(prompt.lower().split()).strip(" .")
    for pat, rel, swap in _RULES:
        m = re.match(pat, p)
        if not m:
            continue
        a, b = m.group("a").strip(), m.group("b").strip()
        if swap:
            a, b = b, a
        if not (_DETERMINER.match(a) and _DETERMINER.match(b)):
            continue
        if _RELATION_WORDS.search(a) or _RELATION_WORDS.search(b):
            return None, f"more than two objects or nested relations ({pat!r} matched with a relation inside a part)"
        return Plan(rel, a, b, pat), f"matched {rel}"
    return None, "no supported two-object relation"


def size_factor(part_prompt: str) -> float:
    words = set(re.findall(r"[a-z]+", part_prompt))
    f = [SIZE_FACTOR[w] for w in SIZE_FACTOR if w in words]
    return f[0] if len(f) == 1 else 1.0


def _bbox(v):
    return v.min(0), v.max(0)


def _affine(scale: float, t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] *= scale
    m[:3, 3] = t
    return m


def _apply(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v @ m[:3, :3].T + m[:3, 3]


def place(plan: Plan, va: np.ndarray, vb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Part vertices (any frame, +Y up) -> (T_a, T_b): 4x4 uniform-scale + translation, part -> scene."""
    def unit(v, prompt):        # scale part to PART_EXTENT * size, centre its bbox at the origin
        lo, hi = _bbox(v)
        s = PART_EXTENT * size_factor(prompt) / max(float((hi - lo).max()), 1e-9)
        return _affine(s, -s * (lo + hi) / 2)
    Ta, Tb = unit(va, plan.a), unit(vb, plan.b)
    la, ha = _bbox(_apply(Ta, va))
    lb, hb = _bbox(_apply(Tb, vb))
    if plan.relation == "on_top_of":            # A's bottom on B's top, x/z centres aligned
        Ta[:3, 3] += np.array([0.0, hb[1] - la[1], 0.0])
    else:                                       # A left of B, GAP between boxes, y/z centres aligned
        Ta[:3, 3] += np.array([-GAP / 2 - ha[0], 0.0, 0.0])
        Tb[:3, 3] += np.array([GAP / 2 - lb[0], 0.0, 0.0])
    lo = np.minimum(_bbox(_apply(Ta, va))[0], _bbox(_apply(Tb, vb))[0])
    hi = np.maximum(_bbox(_apply(Ta, va))[1], _bbox(_apply(Tb, vb))[1])
    s = EXTENT / max(float((hi - lo).max()), 1e-9)
    N = _affine(s, -s * (lo + hi) / 2)
    return N @ Ta, N @ Tb


def assemble(parts: list[dict]) -> dict:
    """Concatenate transformed parts into one scene mesh (parts stay separate meshes; no boolean union)."""
    vs, fs, cs, off = [], [], [], 0
    for p in parts:
        vs.append(p["vertices"])
        fs.append(p["faces"] + off)
        cs.append(p["colors"] if p["colors"] is not None else np.full((len(p["vertices"]), 3), 0.7, np.float32))
        off += len(p["vertices"])
    return {"vertices": np.concatenate(vs).astype(np.float32), "faces": np.concatenate(fs),
            "colors": np.concatenate(cs).astype(np.float32)}


def compose(plan: Plan, generated: list[tuple[str, dict]]) -> dict:
    """Place two generated parts [(generator identity dict, generator output) for a, b] and validate each part and
    the assembled scene. Used by generate_scene and by the E-T3 benchmark on saved generations."""
    from .geometry import validate_mesh
    (ida, oa), (idb, ob) = generated
    for role, o in (("a", oa), ("b", ob)):
        if o["vertices"] is None:
            return {"plan": plan.as_dict(), "status": "part_failed", "failed_part": role, "part_report": o["report"]}
    Ta, Tb = place(plan, oa["vertices"], ob["vertices"])
    roles = plan.as_dict()["roles"]
    parts = []
    for role, text, ident, out, T in (("a", plan.a, ida, oa, Ta), ("b", plan.b, idb, ob, Tb)):
        v = _apply(T, np.asarray(out["vertices"], np.float64)).astype(np.float32)
        parts.append({"prompt": text, "role": roles[role], "generator": ident.get("generator"),
                      "generator_identity": ident, "transform": T.round(9).tolist(), "vertices": v,
                      "faces": out["faces"], "colors": out["colors"], "report": validate_mesh(v, out["faces"]),
                      "generation_report": out["report"]})
    scene = assemble(parts)
    srep = validate_mesh(scene["vertices"], scene["faces"])
    srep.update({"planned": True, "relation": plan.relation, "n_parts": 2,
                 "parts_valid": [bool(p["report"].get("valid")) for p in parts], "units": "normalized_object",
                 "artifact_kind": "triangle_mesh",
                 "seconds": round(sum(float(p["generation_report"].get("seconds") or 0) for p in parts), 3)})
    srep["valid"] = bool(srep.get("valid")) and all(srep["parts_valid"])
    scene["report"] = srep
    return {"plan": plan.as_dict(), "status": "ok", "parts": parts, "scene": scene}


def generate_scene(prompt: str, generator_for, seed: int = 0, cancel=None, progress=None, cache: dict | None = None):
    """Plan, generate both parts (``generator_for(part_prompt) -> Generator``), compose.
    ``cache`` (optional dict) reuses an identical (generator, part prompt, seed) generation.
    -> (result dict or None, reason); result = compose(...) output."""
    plan, reason = parse(prompt)
    if plan is None:
        return None, reason
    generated = []
    for i, (role, text) in enumerate((("a", plan.a), ("b", plan.b))):
        g = generator_for(text)
        key = (g.name, text, seed)
        sub = (lambda fr, st, i=i, role=role: progress((i + fr) / 2 * 0.9, f"part {role}: {st}")) if progress else None
        if cache is not None and key in cache:
            out = cache[key]
        else:
            out = g.generate(text, seed, cancel=cancel, progress=sub)
            if cache is not None:
                cache[key] = out
        generated.append((g.identity, out))
        if out["vertices"] is None:
            break
    if len(generated) < 2:
        return {"plan": plan.as_dict(), "status": "part_failed", "failed_part": "a",
                "part_report": generated[0][1]["report"]}, "part generation failed"
    res = compose(plan, generated)
    if progress:
        progress(0.95, "validating")
    return res, reason if res["status"] == "ok" else "part generation failed"
