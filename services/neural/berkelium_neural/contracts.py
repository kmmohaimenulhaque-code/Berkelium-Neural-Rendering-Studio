"""Typed contracts shared by the renderer, model, dataset, artifact, job and API layers.

Every request is validated here BEFORE any filesystem, model or GPU side effect. Conventions follow
GAUSSIAN_RENDERER_SPEC.md section 4: right-handed +Y-up world, OpenCV camera (+X right, +Y down, +Z forward),
T_cw maps world -> camera, row-major 4x4, pixel centers at (x+0.5, y+0.5), depth = camera Z.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "berkelium-neural/1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ----------------------------------------------------------------------------------------------- camera
class CameraSpec(_Strict):
    """Pinhole camera, OpenCV axes. ``T_cw`` is world->camera (row-major 4x4)."""

    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    T_cw: tuple[tuple[float, float, float, float], ...]
    near: float = Field(default=0.01, gt=0)
    far: float = Field(default=100.0, gt=0)
    model: Literal["pinhole"] = "pinhole"
    convention: Literal["opencv"] = "opencv"

    @field_validator("T_cw")
    @classmethod
    def _pose(cls, m):
        if len(m) != 4 or any(len(r) != 4 for r in m):
            raise ValueError("T_cw must be 4x4")
        if not all(math.isfinite(v) for r in m for v in r):
            raise ValueError("T_cw must be finite")
        if any(abs(a - b) > 1e-9 for a, b in zip(m[3], (0.0, 0.0, 0.0, 1.0), strict=True)):
            raise ValueError("T_cw last row must be [0,0,0,1]")
        R = [r[:3] for r in m[:3]]
        for i in range(3):          # ||R R^T - I||_max <= 1e-5
            for j in range(3):
                d = sum(R[i][k] * R[j][k] for k in range(3)) - (1.0 if i == j else 0.0)
                if abs(d) > 1e-5:
                    raise ValueError("T_cw rotation is not orthonormal (scaled or sheared pose rejected)")
        det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
               + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
        if abs(det - 1.0) > 1e-5:
            raise ValueError("T_cw rotation must be proper (det = +1); reflections rejected")
        return m

    @model_validator(mode="after")
    def _clip(self):
        if not self.near < self.far:
            raise ValueError("require 0 < near < far")
        for v in (self.fx, self.fy, self.cx, self.cy):
            if not math.isfinite(v):
                raise ValueError("intrinsics must be finite")
        return self


# ------------------------------------------------------------------------------------------- rendering
class RenderProfile(str, Enum):
    """Rendering semantics. Comparisons are only meaningful between identical profiles.

    gsplat_v1   AMD gsplat 1.5.3 semantics: alpha=min(0.999, o*exp(-sigma)), skip alpha<1/255 or sigma<0,
                stop EXCLUSIVELY when T*(1-alpha) <= 1e-4, eps2d blur 0.3 px^2, FOV-clipped Jacobian (1.3x),
                opacity-aware 3.33-sigma tile radius.
    berkelium_v02  GAUSSIAN_RENDERER_SPEC.md v0.2: 3-sigma ellipse support (h>9 skipped), alpha cap 0.99, no
                alpha cutoff, threshold-crossing contribution INCLUDED, unclipped Jacobian.
    """

    GSPLAT_V1 = "gsplat_v1"
    BERKELIUM_V02 = "berkelium_v02"


class RenderConfig(_Strict):
    profile: RenderProfile = RenderProfile.GSPLAT_V1
    tile_size: int = Field(default=16, ge=4, le=64)
    eps2d: float = Field(default=0.3, ge=0)
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_gaussians: int = Field(default=1_000_000, ge=1)
    max_pairs: int = Field(default=8_000_000, ge=1)
    outputs: tuple[Literal["rgb", "alpha", "depth"], ...] = ("rgb", "alpha", "depth")


# -------------------------------------------------------------------------------------------- artifacts
class ArtifactKind(str, Enum):
    TRIANGLE_MESH = "triangle_mesh"        # GLB/PLY/OBJ triangle mesh (NOT a CAD model)
    GAUSSIAN_SCENE = "gaussian_scene"      # berkelium-gaussian-soa/1
    IMAGE = "image"                        # rendered or generated image (NOT 3D)
    CHECKPOINT = "checkpoint"              # model / adapter weights
    REPORT = "report"                      # validation / evaluation / benchmark JSON
    DATASET_MANIFEST = "dataset_manifest"


class ArtifactFile(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9._-]{1,128}$")
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class ArtifactRecord(_Strict):
    """Immutable, content-addressed: ``id`` is the sha256 of the canonical JSON of everything else."""

    schema_version: str = SCHEMA_VERSION
    id: str
    kind: ArtifactKind
    files: tuple[ArtifactFile, ...]
    parents: tuple[str, ...] = ()
    producer: dict[str, Any]               # model / renderer identity + revision
    config: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    labels: dict[str, str] = {}            # e.g. {"units": "normalized_object", "confidence": "unverified"}
    created_at: str


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------------- jobs
class JobState(str, Enum):
    """D-010 lifecycle states. Experiment outcomes live in receipts, not here."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CHECKPOINTED = "checkpointed"
    RESUMED = "resumed"


TERMINAL_STATES = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED})


class TextTo3DRequest(_Strict):
    kind: Literal["text_to_3d"] = "text_to_3d"
    prompt: str = Field(min_length=1, max_length=300)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    model: str = Field(default="berkelium-textshape", pattern=r"^[a-z0-9-]{1,64}$")
    adapter: str | None = Field(default=None, pattern=r"^[a-z0-9-]{1,64}$")
    resolution: int = Field(default=48, ge=16, le=128)       # SDF grid resolution for meshing
    gaussian_fit_steps: int = Field(default=0, ge=0, le=2000)  # differentiable refinement (0 = off)

    @field_validator("prompt")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("prompt is empty")
        if any(ord(c) < 32 for c in v):
            raise ValueError("control characters are not allowed")
        return v


class RenderRequest(_Strict):
    kind: Literal["render"] = "render"
    artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    camera: CameraSpec
    config: RenderConfig = RenderConfig()
    backend: Literal["reference_cpu", "gsplat", "berkelium_hip"] = "reference_cpu"


JobRequest = TextTo3DRequest | RenderRequest


class JobRecord(_Strict):
    id: str
    request: dict[str, Any]
    state: JobState
    progress: float = Field(ge=0, le=1)
    stage: str = ""
    artifacts: tuple[str, ...] = ()
    error: dict[str, Any] | None = None
    created_at: str
    updated_at: str


# --------------------------------------------------------------------------------------------- datasets
class DatasetRecord(_Strict):
    """One asset in a dataset manifest. ``family`` groups every variant of an asset so that splits can never
    leak variants of the same underlying asset across train/val/test."""

    asset_id: str
    family: str
    split: Literal["train", "val", "test", "adapter_train", "adapter_test"]
    license: str
    source: str
    asset_sha256: str
    caption: str
    caption_provenance: str
    cameras: tuple[CameraSpec, ...] = ()
    files: dict[str, str] = {}
    attributes: dict[str, Any] = {}


class Measurement(_Strict):
    """One benchmark/evaluation number. Missing measurements are ``status='not_run'`` with ``value=None``,
    never 0."""

    name: str
    value: float | None
    unit: str = ""
    status: Literal["measured", "not_run", "failed", "not_applicable"] = "measured"
    reason: str = ""
