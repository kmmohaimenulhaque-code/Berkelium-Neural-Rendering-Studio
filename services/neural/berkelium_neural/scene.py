"""GaussianScene v1: canonical parameterization + ``berkelium-gaussian-soa/1`` serialization.

Parameterization (GAUSSIAN_RENDERER_SPEC.md section 5): means (world), log_scales, quaternions (w,x,y,z, raw,
normalized at projection), opacity logits (o = sigmoid), degree-0 linear-RGB colour logits (c = sigmoid).
A Gaussian scene is NOT a triangle mesh and is never advertised as one.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

import numpy as np

from .contracts import sha256_hex

FORMAT = "berkelium-gaussian-soa/1"
PLANES = (("means", 3), ("log_scales", 3), ("quats", 4), ("opacity_logits", 1), ("color_logits", 3))
LOG_SCALE_MIN, LOG_SCALE_MAX = float(np.log(1e-5)), float(np.log(1e2))


class SceneError(ValueError):
    pass


@dataclass
class GaussianScene:
    means: np.ndarray            # [N,3] float32
    log_scales: np.ndarray       # [N,3]
    quats: np.ndarray            # [N,4] (w,x,y,z)
    opacity_logits: np.ndarray   # [N]
    color_logits: np.ndarray     # [N,3]
    ids: np.ndarray | None = None  # [N] uint64, unique, immutable tie-break identity
    meta: dict = field(default_factory=dict)  # units, normalization, provenance

    def __post_init__(self):
        n = self.means.shape[0]
        self.means = np.ascontiguousarray(self.means, np.float32).reshape(n, 3)
        self.log_scales = np.ascontiguousarray(self.log_scales, np.float32).reshape(n, 3)
        self.quats = np.ascontiguousarray(self.quats, np.float32).reshape(n, 4)
        self.opacity_logits = np.ascontiguousarray(self.opacity_logits, np.float32).reshape(n)
        self.color_logits = np.ascontiguousarray(self.color_logits, np.float32).reshape(n, 3)
        self.ids = np.arange(n, dtype=np.uint64) if self.ids is None else np.ascontiguousarray(self.ids, np.uint64)

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    def validate(self) -> dict:
        """Raises SceneError on invalid content; returns a diagnostics dict."""
        for name, _ in PLANES:
            a = getattr(self, name)
            if not np.isfinite(a).all():
                raise SceneError(f"{name} contains non-finite values")
        qn = np.linalg.norm(self.quats, axis=1)
        if (qn < 1e-8).any():
            raise SceneError(f"INVALID_QUATERNION at indices {np.nonzero(qn < 1e-8)[0][:10].tolist()}")
        if self.ids.shape[0] != self.n or len(np.unique(self.ids)) != self.n:
            raise SceneError("gaussian ids must be unique, one per Gaussian")
        clamped = int(((self.log_scales < LOG_SCALE_MIN) | (self.log_scales > LOG_SCALE_MAX)).any(1).sum())
        lo, hi = (self.means.min(0).tolist(), self.means.max(0).tolist()) if self.n else ([0] * 3, [0] * 3)
        return {"n": self.n, "log_scale_clamped": clamped, "bounds_min": lo, "bounds_max": hi}

    # ----------------------------------------------------------------------------------- activations
    def scales(self) -> np.ndarray:
        return np.exp(np.clip(self.log_scales, LOG_SCALE_MIN, LOG_SCALE_MAX))

    def opacities(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.opacity_logits))

    def colors(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.color_logits))

    # -------------------------------------------------------------------------------- serialization
    def to_bytes(self) -> tuple[bytes, dict]:
        """Little-endian SoA payload + manifest (offsets, sizes, sha256 per plane, counts, types, basis)."""
        self.validate()
        buf, planes, off = io.BytesIO(), [], 0
        arrays = [(name, getattr(self, name), "float32") for name, _ in PLANES] + [("ids", self.ids, "uint64")]
        for name, a, dtype in arrays:
            raw = np.ascontiguousarray(a, dtype=np.dtype(dtype).newbyteorder("<")).tobytes()
            pad = (-off) % 256                      # each plane starts on a 256-byte boundary
            buf.write(b"\0" * pad)
            off += pad
            planes.append({"name": name, "dtype": dtype, "shape": list(a.shape), "offset": off, "bytes": len(raw),
                           "sha256": sha256_hex(raw)})
            buf.write(raw)
            off += len(raw)
        payload = buf.getvalue()
        manifest = {"format": FORMAT, "endian": "little", "n": self.n, "planes": planes,
                    "payload_sha256": sha256_hex(payload), "payload_bytes": len(payload),
                    "parameterization": {"scale": "log", "rotation": "quat_wxyz_raw", "opacity": "logit",
                                         "color": "linear_rgb_logit_deg0"},
                    "coordinate_system": {"world": "right_handed_y_up", "camera": "opencv"},
                    "meta": self.meta}
        return payload, manifest

    @classmethod
    def from_bytes(cls, payload: bytes, manifest: dict) -> GaussianScene:
        if manifest.get("format") != FORMAT:
            raise SceneError(f"unsupported format {manifest.get('format')!r}")
        if len(payload) != manifest["payload_bytes"] or sha256_hex(payload) != manifest["payload_sha256"]:
            raise SceneError("payload truncated or corrupted (size/sha256 mismatch)")
        n, arrs, spans = int(manifest["n"]), {}, []
        expected = {**{k: (n, c) if c > 1 else (n,) for k, c in PLANES}, "ids": (n,)}
        for p in manifest["planes"]:
            if p["name"] not in expected or tuple(p["shape"]) != expected[p["name"]]:
                raise SceneError(f"unexpected plane {p['name']} {p['shape']}")
            end = p["offset"] + p["bytes"]
            if p["offset"] < 0 or end > len(payload):
                raise SceneError(f"plane {p['name']} out of bounds")
            if any(not (end <= s or p["offset"] >= e) for s, e in spans):
                raise SceneError(f"plane {p['name']} overlaps another plane")
            spans.append((p["offset"], end))
            raw = payload[p["offset"]:end]
            if sha256_hex(raw) != p["sha256"]:
                raise SceneError(f"plane {p['name']} checksum mismatch")
            dt = np.dtype(p["dtype"]).newbyteorder("<")
            if int(np.prod(p["shape"])) * dt.itemsize != len(raw):
                raise SceneError(f"plane {p['name']} size mismatch")
            arrs[p["name"]] = np.frombuffer(raw, dtype=dt).reshape(p["shape"]).astype(dt.newbyteorder("="))
        if set(arrs) != set(expected):
            raise SceneError(f"missing planes {sorted(set(expected) - set(arrs))}")
        s = cls(**{k: arrs[k] for k, _ in PLANES}, ids=arrs["ids"], meta=manifest.get("meta", {}))
        s.validate()
        return s

    def to_3dgs_ply(self) -> bytes:
        """Graphdeco-convention Gaussian PLY (DC-only SH) for third-party viewers. f_dc = (c-0.5)/C0.
        Declared as a Gaussian PLY, never a triangle mesh."""
        C0 = 0.28209479177387814
        c = self.colors()
        cols = [self.means, np.zeros_like(self.means), (c - 0.5) / C0, self.opacity_logits[:, None],
                np.clip(self.log_scales, LOG_SCALE_MIN, LOG_SCALE_MAX), self.quats]
        names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
                 "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        data = np.concatenate(cols, 1).astype("<f4")
        head = "ply\nformat binary_little_endian 1.0\ncomment berkelium gaussian scene (NOT a mesh)\n" \
               f"element vertex {self.n}\n" + "".join(f"property float {k}\n" for k in names) + "end_header\n"
        return head.encode() + data.tobytes()


def manifest_bytes(manifest: dict) -> bytes:
    return json.dumps(manifest, indent=2, sort_keys=True).encode()
