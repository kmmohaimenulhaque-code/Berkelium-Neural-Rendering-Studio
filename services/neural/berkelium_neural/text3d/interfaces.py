"""Typed ML interfaces (distinct from the software-integration adapters in render/ and service/).

Text3DModel      text encoding + 3D generation (+ optional reconstruction from images)
TrainingAdapter  a model's ACTUALLY SUPPORTED fine-tuning mechanism (never a universal format)

Implemented: BerkeliumTextShape (text -> SDF -> mesh) with LoRA+token-embedding-delta training adapter.
Not integrated (each needs a weight download that has not been approved; listed so nobody assumes they exist):
  Shap-E (MIT, openai/shap-e)        text->implicit; fine-tuning would need its latent encoder (separate weights)
  Point-E base40M-textvec (MIT)      text->point cloud diffusion; fine-tunable on point clouds directly
  TripoSR (MIT, stabilityai)         single image->mesh reconstruction; NOT text-conditioned by itself
Decision D-006 (Z-Image -> TripoSR) remains a candidate path; it is image-conditioned 3D, not trained here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np


@dataclass
class Generated3D:
    vertices: np.ndarray | None
    faces: np.ndarray | None
    colors: np.ndarray | None
    report: dict


@runtime_checkable
class Text3DModel(Protocol):
    name: str
    capabilities: dict          # {"text_to_3d": bool, "image_to_3d": bool, "output": "triangle_mesh" | ...}

    def encode_text(self, prompts: list[str]): ...

    def generate(self, prompt: str, seed: int = 0, resolution: int = 48) -> Generated3D: ...


@runtime_checkable
class TrainingAdapter(Protocol):
    mechanism: str              # e.g. "lora+token_embedding_delta"

    def attach(self, model) -> list: ...           # returns trainable parameters; base frozen

    def state(self, model) -> dict: ...

    def save(self, model, path: Path, base_sha256: str) -> None: ...


class BerkeliumTextShape:
    """Text3DModel implementation over berkelium_neural.text3d.model.TextShapeModel."""

    capabilities: ClassVar[dict] = {"text_to_3d": True, "image_to_3d": False, "output": "triangle_mesh",
                    "units": "normalized_object", "fine_tuning": ["full", "lora+token_embedding_delta"]}

    def __init__(self, checkpoint: Path, adapter: Path | None = None):
        from .train import load_model
        self.model = load_model(Path(checkpoint), Path(adapter) if adapter else None)
        self.name = "berkelium-textshape" + ("+adapter" if adapter else "")

    def encode_text(self, prompts):
        import torch
        with torch.no_grad():
            return self.model.encode(list(prompts))

    def generate(self, prompt, seed=0, resolution=48) -> Generated3D:
        from .generate import generate_mesh
        o = generate_mesh(self.model, prompt, resolution)     # deterministic: seed recorded, not used
        o["report"]["seed"] = seed
        return Generated3D(o["vertices"], o["faces"], o["colors"], o["report"])


class LoRAAdapter:
    """TrainingAdapter for TextShapeModel: LoRA on LORA_TARGETS + embedding deltas for named new tokens."""

    mechanism = "lora+token_embedding_delta"

    def __init__(self, r: int = 8, alpha: float = 16.0, new_tokens: tuple[str, ...] = ()):
        self.r, self.alpha, self.new_tokens = r, alpha, list(new_tokens)

    def attach(self, model):
        from .model import apply_adapter
        return apply_adapter(model, self.r, self.alpha, self.new_tokens)

    def state(self, model):
        from .model import adapter_state
        return adapter_state(model)

    def save(self, model, path, base_sha256):
        import torch
        torch.save({"state": self.state(model), "adapter_config": model.adapter_config,
                    "base_checkpoint_sha256": base_sha256}, path)
