"""Studio ``cadquery`` backend: text → E-C2 compiler (Qwen3 + CadQuery LoRA) → CadQuery script → sandbox →
STEP + GLB → Berkelium core validation (repair ≤ 2 rounds) → mesh → the existing Gaussian fit / render.

The compiler is reached through one of, in order:

1. an injected ``chat(batch_of_messages) -> list[str]`` (tests, embedding apps);
2. ``BERKELIUM_CAD_URL``: an OpenAI-compatible endpoint (vLLM ``serve`` with the LoRA, or an HF Inference
   Endpoint on the merged model); ``BERKELIUM_CAD_MODEL`` names the served model, the key comes from
   ``BERKELIUM_CAD_KEY`` or ``HF_TOKEN``;
3. ``BERKELIUM_CAD_BASE`` (+ optional ``BERKELIUM_CAD_ADAPTER``) on a local GPU: vLLM if it starts, else HF
   transformers. A merged model is ``BERKELIUM_CAD_BASE`` alone.

Opt-in until D-034 passes: select generator ``cadquery`` or set ``BERKELIUM_CAD_BACKEND=cadquery`` for ``cad``.
The model only proposes; the sandbox executes and OCCT through the core measures and validates."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

NAME = "cadquery"
_CQ_DIR = Path(os.environ.get("BERKELIUM_CQ_DIR", Path(__file__).resolve().parents[4] / "cadquery"))


class CadQueryUnavailable(RuntimeError):
    pass


def _cq_modules():
    if str(_CQ_DIR) not in sys.path:
        sys.path.insert(0, str(_CQ_DIR))
    import cqformat
    import cqpipeline
    import sandbox
    return cqformat, cqpipeline, sandbox


def glb_mesh(glb: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Positions + triangle indices from the sandbox's single-primitive GLB."""
    magic, _, _ = struct.unpack_from("<III", glb, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB")
    jlen, _ = struct.unpack_from("<II", glb, 12)
    gltf = json.loads(glb[20:20 + jlen])
    blen, _ = struct.unpack_from("<II", glb, 20 + jlen)
    binc = glb[28 + jlen:28 + jlen + blen]
    acc, views = gltf["accessors"], gltf["bufferViews"]

    def arr(i, dt, w):
        v = views[acc[i]["bufferView"]]
        a = np.frombuffer(binc, dtype=dt, count=acc[i]["count"] * w, offset=v.get("byteOffset", 0))
        return a.reshape(-1, w) if w > 1 else a
    prim = gltf["meshes"][0]["primitives"][0]
    return arr(prim["attributes"]["POSITION"], np.float32, 3).copy(), arr(prim["indices"], np.uint32, 1).reshape(-1, 3).astype(np.int64)


class CadQueryGenerator:
    name = NAME

    def __init__(self, chat: Callable[[list[list[dict]]], list[str]] | None = None, rounds: int | None = None):
        self._chat = chat
        self._backend = None
        self.rounds = rounds

    # ------------------------------------------------------------------ compiler
    def _resolve(self) -> tuple[Callable | None, str]:
        if self._chat is not None:
            return self._chat, "injected"
        if self._backend is not None:
            return self._backend.chat, self._backend.name
        if str(_CQ_DIR) not in sys.path:
            sys.path.insert(0, str(_CQ_DIR))
        from llm import EndpointBackend, make_backend
        if os.environ.get("BERKELIUM_CAD_URL"):
            self._backend = EndpointBackend(os.environ["BERKELIUM_CAD_URL"], os.environ.get("BERKELIUM_CAD_MODEL", NAME))
            return self._backend.chat, "endpoint"
        base = os.environ.get("BERKELIUM_CAD_BASE")
        if base:
            try:
                import torch
                if not torch.cuda.is_available():
                    return None, "BERKELIUM_CAD_BASE is set but no GPU (CUDA/ROCm) is available"
            except ImportError:
                return None, "BERKELIUM_CAD_BASE is set but torch is not installed"
            import pins
            rev = pins.MODELS.get(base, {}).get("revision", os.environ.get("BERKELIUM_CAD_REVISION", "main"))
            self._backend, _ = make_backend(base, rev, os.environ.get("BERKELIUM_CAD_ADAPTER"))
            return self._backend.chat, f"local-{self._backend.name}"
        return None, ("no CadQuery compiler: set BERKELIUM_CAD_URL (endpoint serving the E-C2 model) or "
                      "BERKELIUM_CAD_BASE [+ BERKELIUM_CAD_ADAPTER] on a GPU host")

    def available(self) -> tuple[bool, str]:
        try:
            import cadquery  # noqa: F401
            import berkelium  # noqa: F401
        except ImportError as e:
            return False, f"{e.name} not installed (pip install cadquery and the pinned Berkelium core)"
        if not (_CQ_DIR / "sandbox.py").exists():
            return False, f"sandbox not found at {_CQ_DIR}"
        if self._chat is None and not (os.environ.get("BERKELIUM_CAD_URL") or os.environ.get("BERKELIUM_CAD_BASE")):
            return False, "no compiler configured (BERKELIUM_CAD_URL or BERKELIUM_CAD_BASE)"
        return True, ""

    @property
    def identity(self) -> dict:
        return {"generator": NAME, "compiler": os.environ.get("BERKELIUM_CAD_MODEL") or
                os.environ.get("BERKELIUM_CAD_ADAPTER") or os.environ.get("BERKELIUM_CAD_BASE") or "unconfigured",
                "format": "cq-v1", "decision": "D-034 (opt-in until it passes)", "authority": "sandbox + berkelium core"}

    # ------------------------------------------------------------------ generate
    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict:
        ok, why = self.available()
        if not ok and self._chat is None:
            raise CadQueryUnavailable(why)
        cqformat, cqpipeline, sandbox = _cq_modules()
        chat, how = self._resolve()
        if chat is None:
            raise CadQueryUnavailable(how)
        t0 = time.time()
        if progress:
            progress(0.1, "compiling text to a CadQuery script")
        rec = cqpipeline.compile_text(prompt, chat, None,
                                      rounds=cqpipeline.REPAIR_ROUNDS if self.rounds is None else self.rounds)
        if not rec["ok"] or not rec.get("_glb"):
            raise CadQueryUnavailable(f"no valid geometry after {len(rec['history'])} attempt(s): "
                                      f"[{rec['stage']}] {rec['error'][:500]}")
        if progress:
            progress(0.6, "meshing the core-validated solid")
        verts, faces = glb_mesh(rec["_glb"])
        verts = verts[:, [0, 2, 1]] * np.array([1, 1, -1], np.float32)       # CadQuery Z-up mm -> Studio Y-up
        ext = float(np.ptp(verts, 0).max()) or 1.0
        verts_n = (verts - (verts.min(0) + verts.max(0)) / 2) / ext
        header = rec.get("header") or {}
        step = rec["_step"]
        report = {"valid": True, "source": how, "backend": NAME, "summary": header.get("summary"),
                  "intent": header, "open_questions": header.get("open_questions", []),
                  "assumptions": header.get("assumptions", []), "script": rec.get("code"),
                  "completion": rec["completion"], "attempts": rec["history"],
                  "validation": {"summary": rec["core"]["summary"], "counts": {}, "failures": []},
                  "measurements": rec["core"]["measurements"], "artifacts": [{"name": "result.step", "sha256":
                  hashlib.sha256(step).hexdigest(), "bytes": len(step)}], "record_hash": hashlib.sha256(step).hexdigest(),
                  "proposal_hash": hashlib.sha256((rec.get("code") or "").encode()).hexdigest(),
                  "scale_mm_per_unit": ext, "seconds": round(time.time() - t0, 3), "net_isolation": rec.get("net_isolation"),
                  "note": "the model only proposed the script; the sandbox executed it and the Berkelium core (OCCT) "
                          "measured and validated the solid"}
        return {"vertices": verts_n.astype(np.float32), "faces": faces, "colors": None, "report": report,
                "step": step}
