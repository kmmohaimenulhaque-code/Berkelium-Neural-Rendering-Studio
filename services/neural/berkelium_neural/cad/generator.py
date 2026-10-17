"""Studio ``cad`` generator: text -> design compiler -> DesignProposal -> Berkelium core -> exact geometry -> mesh.

The compiler is the E-C1 model (Qwen3-8B + cad LoRA). It is reached through one of, in order:

1. an injected callable ``compiler(text) -> completion`` (tests, embedding apps);
2. an OpenAI-compatible endpoint: ``BERKELIUM_CAD_URL`` (+ ``BERKELIUM_CAD_MODEL``, ``BERKELIUM_CAD_KEY``), e.g. vLLM
   serving Qwen3-8B with the LoRA;
3. a local CUDA GPU with the adapter at ``BERKELIUM_CAD_ADAPTER`` (NF4, as evaluated).

A prompt that already *is* a DesignProposal JSON skips the model. The core is the only authority: the mesh, the
measurements and the validation summary come from ``berkelium.pipeline.run``; nothing the model wrote is reported
as measured. A proposal that fails is returned as a failure with the core's digest, never patched silently."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable

import numpy as np

from . import CORE_COMMIT
from .prompts import PROMPT_VERSION, extract_json, messages

NAME = "cad"


class CadUnavailable(RuntimeError):
    pass


def core_available() -> tuple[bool, str]:
    try:
        import berkelium  # noqa: F401
        import manifold3d  # noqa: F401
    except ImportError as e:
        return False, f"Berkelium core not installed ({e.name}): pip install -r configs/environments/berkelium-core.txt"
    return True, ""


def _catalogue() -> str:
    from berkelium.ai.prompts import cem_catalogue
    from berkelium.cem.protocol import default_registry
    return cem_catalogue(default_registry())


def endpoint_compiler(url: str, model: str, key: str | None = None, timeout: float = 300) -> Callable[[str], str]:
    import httpx

    def call(text: str) -> str:
        r = httpx.post(url.rstrip("/") + "/chat/completions", timeout=timeout,
                       headers={"Authorization": f"Bearer {key}"} if key else {},
                       json={"model": model, "messages": messages(text, _catalogue()), "temperature": 0,
                             "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    return call


def local_compiler(adapter: str) -> Callable[[str], str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    base, rev = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
    tok = AutoTokenizer.from_pretrained(base, revision=rev)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(
        base, revision=rev, quantization_config=bnb, dtype=torch.float16, device_map={"": 0}), adapter).eval()
    end = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]

    def call(text: str) -> str:
        p = tok.apply_chat_template(messages(text, _catalogue()), tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to(0)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=2048, do_sample=False, eos_token_id=end)
        return tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return call


class CadGenerator:
    name = NAME

    def __init__(self, compiler: Callable[[str], str] | None = None, artifacts_dir: str | None = None):
        self._compiler = compiler
        self.artifacts_dir = artifacts_dir

    def _resolve_compiler(self) -> tuple[Callable[[str], str] | None, str]:
        if self._compiler is not None:
            return self._compiler, "injected"
        if os.environ.get("BERKELIUM_CAD_URL"):
            return endpoint_compiler(os.environ["BERKELIUM_CAD_URL"], os.environ.get("BERKELIUM_CAD_MODEL", "cad"),
                                     os.environ.get("BERKELIUM_CAD_KEY")), "endpoint"
        adapter = os.environ.get("BERKELIUM_CAD_ADAPTER")
        if adapter:
            try:
                import torch
                if torch.cuda.is_available():
                    return local_compiler(adapter), "local-gpu"
            except ImportError:
                pass
        return None, ("no compiler: set BERKELIUM_CAD_URL (OpenAI-compatible endpoint serving Qwen3-8B + the cad "
                      "LoRA) or BERKELIUM_CAD_ADAPTER on a CUDA host")

    def available(self) -> tuple[bool, str]:
        ok, why = core_available()
        if not ok:
            return False, why
        if self._compiler is not None or os.environ.get("BERKELIUM_CAD_URL") or os.environ.get("BERKELIUM_CAD_ADAPTER"):
            return True, ""
        return True, "only DesignProposal JSON prompts (no compiler configured)"

    @property
    def identity(self) -> dict:
        return {"generator": NAME, "prompt_version": PROMPT_VERSION, "core_commit": CORE_COMMIT,
                "compiler": "Qwen/Qwen3-8B@b968826d + cad LoRA (E-C1)", "authority": "berkelium core"}

    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict:
        ok, why = core_available()
        if not ok:
            raise CadUnavailable(why)
        from berkelium.geometry.manifold_backend import ManifoldBackend
        from berkelium.pipeline import ArtifactStore, run
        t0 = time.time()
        proposal = extract_json(prompt) if prompt.lstrip().startswith("{") else None
        source, completion = "proposal_json", None
        if proposal is None:
            compiler, how = self._resolve_compiler()
            if compiler is None:
                raise CadUnavailable(how)
            if progress:
                progress(0.1, "compiling text to a design proposal")
            completion = compiler(prompt)
            proposal, source = extract_json(completion), how
            if proposal is None:
                raise CadUnavailable("the compiler did not return a JSON design proposal")
        if progress:
            progress(0.5, "realising geometry in the Berkelium core")
        store = ArtifactStore(self.artifacts_dir) if self.artifacts_dir else None
        res = run(proposal, realize=True, artifacts=store)
        if res.record is None:
            raise CadUnavailable("design proposal is not schema-valid: " +
                                 json.dumps(res.schema_errors[:3], default=str)[:600])
        rec = res.record
        v = rec.evaluation.validation
        failures = [f"L{r.level} {r.target}: {r.message}" for r in v.results if r.status in ("fail", "error")]
        mb, solid = ManifoldBackend(), None
        for rc in rec.realized:
            if rc.geometry is not None:
                for body in mb.execute(rc.geometry).bodies.values():
                    solid = body if solid is None else solid + body
        if solid is None:
            raise CadUnavailable("no geometry realised: " + "; ".join(failures[:5]))
        mesh = solid.to_mesh()
        verts = np.asarray(mesh.vert_properties, dtype=np.float32)[:, :3]
        faces = np.asarray(mesh.tri_verts, dtype=np.int64)
        verts = verts[:, [0, 2, 1]] * np.array([1, 1, -1], np.float32)      # core Z-up mm -> Studio Y-up
        ext = float(np.ptp(verts, 0).max()) or 1.0
        verts_n = (verts - (verts.min(0) + verts.max(0)) / 2) / ext            # Studio convention: unit extent
        report = {"valid": not failures, "source": source, "proposal": proposal, "completion": completion,
                  "intent": proposal.get("intent", {}),
                  "open_questions": proposal.get("intent", {}).get("open_questions", []),
                  "validation": {"summary": v.summary, "counts": v.counts, "failures": failures[:20]},
                  "measurements": [{"name": m.name, "target": m.target,
                                    "value": m.value.model_dump() if m.value else None, "vector": m.vector}
                                   for m in rec.evaluation.measurements],
                  "artifacts": [a.model_dump() for a in rec.evaluation.artifacts],
                  "record_hash": rec.content_hash, "proposal_hash": rec.proposal_hash,
                  "scale_mm_per_unit": ext, "seconds": round(time.time() - t0, 3),
                  "note": "geometry, measurements and validation are computed by the Berkelium core; "
                          "the model only proposed the design"}
        return {"vertices": verts_n.astype(np.float32), "faces": faces, "colors": None, "report": report}
