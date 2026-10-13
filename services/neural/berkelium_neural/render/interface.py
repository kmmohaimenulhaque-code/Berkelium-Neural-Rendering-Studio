"""Typed renderer interface shared by the CPU reference, the AMD gsplat adapter and the Berkelium HIP extension.

A renderer maps (GaussianTensors, CameraSpec, RenderConfig) -> RenderOutput. Every backend reports what it
actually executed (backend, device, deterministic flag, profile) so results are never mislabelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from ..contracts import CameraSpec, RenderConfig
from ..scene import LOG_SCALE_MAX, LOG_SCALE_MIN, GaussianScene


class RendererError(RuntimeError):
    """Typed renderer failure. ``code`` follows the spec error list (UNSUPPORTED_FEATURE, RESOURCE_LIMIT,
    DEVICE_UNAVAILABLE, NUMERIC_FAILURE, INVALID_SCENE, ...)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class GaussianTensors:
    means: torch.Tensor           # [N,3]
    log_scales: torch.Tensor      # [N,3]
    quats: torch.Tensor           # [N,4] (w,x,y,z), raw
    opacity_logits: torch.Tensor  # [N]
    color_logits: torch.Tensor    # [N,3]

    @classmethod
    def from_scene(cls, s: GaussianScene, device="cpu", dtype=torch.float32, requires_grad=False):
        def t(a):
            return torch.tensor(np.asarray(a), device=device, dtype=dtype, requires_grad=requires_grad)
        return cls(t(s.means), t(s.log_scales), t(s.quats), t(s.opacity_logits), t(s.color_logits))

    def params(self) -> list[torch.Tensor]:
        return [self.means, self.log_scales, self.quats, self.opacity_logits, self.color_logits]

    def activated(self):
        """(means, quats_normalized, scales, opacities, colors) with the canonical activations."""
        qn = self.quats.norm(dim=-1, keepdim=True)
        if (qn < 1e-8).any():
            raise RendererError("INVALID_QUATERNION", "quaternion norm < 1e-8")
        scales = torch.exp(self.log_scales.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX))
        return self.means, self.quats / qn, scales, torch.sigmoid(self.opacity_logits), torch.sigmoid(self.color_logits)


@dataclass
class RenderOutput:
    rgb: torch.Tensor        # [H,W,3] linear RGB (background composited)
    alpha: torch.Tensor      # [H,W]
    depth: torch.Tensor      # [H,W] expected camera-Z depth (profile-defined invalid value where alpha ~ 0)
    meta: dict = field(default_factory=dict)
    aux: dict = field(default_factory=dict)   # backend intermediates (gsplat layout) for parity/backward tests


class Renderer(Protocol):
    name: str

    def capabilities(self) -> dict: ...

    def prepare(self, cfg: RenderConfig) -> str:
        """Check the backend is actually available (raise RendererError otherwise); return its torch device."""
        ...

    def render(self, g: GaussianTensors, cam: CameraSpec, cfg: RenderConfig) -> RenderOutput: ...


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Hamilton (w,x,y,z), q already normalized -> [...,3,3] active rotation local->world."""
    w, x, y, z = q.unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return R.reshape(q.shape[:-1] + (3, 3))


def covariance(quats_n: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    R = quat_to_rotmat(quats_n)
    M = R * scales[..., None, :]
    return M @ M.transpose(-1, -2)
