"""Prompt -> triangle mesh (+ vertex colours) -> optional Gaussian scene. Real 3D output, validated."""

from __future__ import annotations

import io
import math
import time

import numpy as np
import torch

from ..scene import GaussianScene
from .geometry import grid_points, mesh_from_sdf_grid, sample_surface, validate_mesh
from .model import TextShapeModel, tokenize


@torch.no_grad()
def generate_mesh(model: TextShapeModel, prompt: str, resolution: int = 48, chunk: int = 65536, cancel=None):
    """Returns dict(vertices, faces, colors, report). ``cancel()`` is polled between chunks."""
    t0 = time.time()
    ids, unknown = tokenize(prompt, model.vocab)
    z = model.encode([prompt])
    pts = torch.from_numpy(grid_points(resolution))
    sd = torch.empty(len(pts))
    for s in range(0, len(pts), chunk):
        if cancel and cancel():
            raise InterruptedError("cancelled")
        sd[s:s + chunk] = model.decoder(pts[None, s:s + chunk], z)[0][0]
    vol = sd.numpy().reshape((resolution,) * 3)
    v, f = mesh_from_sdf_grid(vol)
    col = None
    if v is not None:
        _, logit = model.decoder(torch.from_numpy(v)[None], z)
        col = torch.sigmoid(logit[0]).numpy().astype(np.float32)
    rep = validate_mesh(v, f)
    rep.update({"prompt": prompt, "tokens": len(ids), "unknown_words": unknown, "resolution": resolution,
                "sdf_min": float(vol.min()), "sdf_max": float(vol.max()), "seconds": round(time.time() - t0, 3),
                "artifact_kind": "triangle_mesh", "units": "normalized_object"})
    return {"vertices": v, "faces": f, "colors": col, "report": rep}


def mesh_to_glb(v, f, colors) -> bytes:
    import trimesh
    vc = None if colors is None else (np.clip(colors, 0, 1) * 255).round().astype(np.uint8)
    m = trimesh.Trimesh(v, f, vertex_colors=vc, process=False)
    return m.export(file_type="glb")


def mesh_to_ply(v, f, colors) -> bytes:
    import trimesh
    vc = None if colors is None else (np.clip(colors, 0, 1) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    trimesh.Trimesh(v, f, vertex_colors=vc, process=False).export(buf, file_type="ply")
    return buf.getvalue()


def _quat_from_normal(n: np.ndarray) -> np.ndarray:
    """Rotation taking local +Z to each normal (w,x,y,z)."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(np.broadcast_to(z, n.shape), n)
    s = np.linalg.norm(axis, axis=1)
    c = n @ z
    ang = np.arctan2(s, c)
    axis = np.where(s[:, None] > 1e-8, axis / np.maximum(s[:, None], 1e-12), np.array([1.0, 0, 0]))
    return np.concatenate([np.cos(ang / 2)[:, None], axis * np.sin(ang / 2)[:, None]], 1)


def mesh_to_gaussians(v, f, colors, n: int = 4000, seed: int = 0, thickness: float = 0.004) -> GaussianScene:
    """Surface-aligned Gaussian initialisation: flat Gaussians tangent to the mesh (disc radius from area density)."""
    pts, nrm, col = sample_surface(v, f, n, seed, colors)
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()
    r = 0.75 * math.sqrt(area / n)
    ls = np.log(np.tile([r, r, thickness], (n, 1)))
    col = np.clip(col if col is not None else np.full((n, 3), 0.7), 1e-4, 1 - 1e-4)
    return GaussianScene(pts, ls, _quat_from_normal(nrm.astype(np.float64)), np.full(n, 3.0), np.log(col / (1 - col)),
                         meta={"units": "normalized_object", "init": "mesh_surface_samples", "n": n, "seed": seed})
