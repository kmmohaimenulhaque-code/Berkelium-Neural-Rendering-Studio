"""Meshing, mesh validation, surface sampling, a CPU z-buffer mesh renderer, and geometry/image metrics.

All metrics are computed against INDEPENDENT ground truth (analytic SDF meshes from shapes.py). Agreement with a
model's own renders is never used as evidence of geometric correctness.
"""

from __future__ import annotations

import math

import numpy as np

from ..camera import T_cw
from ..contracts import CameraSpec

BOUND = 1.0   # meshing / evaluation cube [-1,1]^3 (normalized_object units)


def grid_points(res: int, bound: float = BOUND) -> np.ndarray:
    a = np.linspace(-bound, bound, res, dtype=np.float32)
    return np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)


def mesh_from_sdf_grid(vol: np.ndarray, bound: float = BOUND):
    """Marching cubes at level 0 -> (vertices [V,3], faces [F,3]) in object units, outward normals.
    Returns (None, None) if the zero level set does not cross the grid (no surface)."""
    from skimage.measure import marching_cubes
    if not (vol.min() < 0 < vol.max()):
        return None, None
    res = vol.shape[0]
    step = 2 * bound / (res - 1)
    v, f, _, _ = marching_cubes(vol.astype(np.float32), level=0.0, spacing=(step,) * 3)
    v = v - bound
    return clean_mesh(v.astype(np.float32), f.astype(np.int64))[:2]   # skimage winding is outward for SDF (<0 in)


def clean_mesh(v: np.ndarray, f: np.ndarray, min_area: float = 1e-12):
    """Drop zero-area faces (marching cubes emits them at sharp apexes) and unreferenced vertices.
    Returns (v, f, n_removed_faces)."""
    a = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    keep = a >= min_area
    f = f[keep]
    used = np.unique(f)
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f], int((~keep).sum())


def validate_mesh(v: np.ndarray | None, f: np.ndarray | None, bound: float = BOUND) -> dict:
    """Validity report (never raises). ``valid`` = finite, non-empty, inside bounds, no degenerate faces."""
    if v is None or f is None or len(f) == 0:
        return {"valid": False, "reason": "empty_mesh", "n_vertices": 0, "n_faces": 0}
    import trimesh
    m = trimesh.Trimesh(v, f, process=False)
    areas = m.area_faces
    degenerate = int((areas < 1e-12).sum())
    comps = m.split(only_watertight=False)
    finite = bool(np.isfinite(v).all())
    in_bounds = bool((np.abs(v) <= bound + 1e-6).all())
    edges = m.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    rep = {"n_vertices": len(v), "n_faces": len(f), "finite": finite, "in_bounds": in_bounds,
           "degenerate_faces": degenerate, "components": len(comps),
           "watertight": bool(m.is_watertight), "winding_consistent": bool(m.is_winding_consistent),
           "edge_manifold": bool((counts <= 2).all()), "euler_number": int(m.euler_number),
           "surface_area": float(m.area), "volume": float(m.volume) if m.is_watertight else None,
           "bounds_min": v.min(0).round(4).tolist(), "bounds_max": v.max(0).round(4).tolist()}
    rep["valid"] = finite and in_bounds and degenerate == 0
    return rep


def sample_surface(v, f, n: int, seed: int = 0, vertex_colors: np.ndarray | None = None):
    """Area-weighted surface samples -> (points [n,3], normals [n,3], colors [n,3] or None)."""
    rng = np.random.default_rng(seed)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    idx = rng.choice(len(f), size=n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    bary = np.stack([1 - u - w, u, w], 1)
    pts = (tri[idx] * bary[:, :, None]).sum(1)
    nrm = cross[idx] / np.maximum(np.linalg.norm(cross[idx], axis=1, keepdims=True), 1e-12)
    col = None if vertex_colors is None else (vertex_colors[f[idx]] * bary[:, :, None]).sum(1)
    return pts.astype(np.float32), nrm.astype(np.float32), col


def geometry_metrics(pred_v, pred_f, gt_v, gt_f, n: int = 20000, seed: int = 0, taus=(0.02, 0.05)) -> dict:
    """Chamfer (L1 mean of NN distances, and squared-L2), normal consistency, F-score@tau, all vs GT mesh."""
    from scipy.spatial import cKDTree
    if pred_v is None or pred_f is None or len(pred_f) == 0:
        return {"chamfer_l1": None, "chamfer_l2": None, "normal_consistency": None,
                **{f"fscore@{t}": 0.0 for t in taus}, "status": "no_prediction"}
    p, pn, _ = sample_surface(pred_v, pred_f, n, seed)
    g, gn, _ = sample_surface(gt_v, gt_f, n, seed + 1)
    dp, ip = cKDTree(g).query(p)
    dg, ig = cKDTree(p).query(g)
    nc = 0.5 * (np.abs((pn * gn[ip]).sum(1)).mean() + np.abs((gn * pn[ig]).sum(1)).mean())
    out = {"chamfer_l1": float(0.5 * (dp.mean() + dg.mean())), "chamfer_l2": float(0.5 * ((dp**2).mean() +
           (dg**2).mean())), "normal_consistency": float(nc), "status": "measured"}
    for t in taus:
        prec, rec = (dp < t).mean(), (dg < t).mean()
        out[f"fscore@{t}"] = float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0
    return out


# ------------------------------------------------------------------------------------------- renderer
def render_mesh(v, f, vertex_colors, cam: CameraSpec, background=(1.0, 1.0, 1.0), shade: bool = True):
    """Deterministic CPU z-buffer rasterizer (pixel centres, perspective-correct barycentrics, Lambert + ambient).
    Returns (rgb [H,W,3] float32, mask [H,W] bool, depth [H,W] camera-Z, 0 where empty)."""
    H, W = cam.height, cam.width
    img = np.empty((H, W, 3), np.float32)
    img[:] = background
    zbuf = np.full((H, W), np.inf, np.float32)
    if v is None or f is None or len(f) == 0:
        return img, np.zeros((H, W), bool), np.zeros((H, W), np.float32)
    T = T_cw(cam)
    pc = v @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([cam.fx * pc[:, 0] / z + cam.cx, cam.fy * pc[:, 1] / z + cam.cy], 1)
    tri_n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    tri_n /= np.maximum(np.linalg.norm(tri_n, axis=1, keepdims=True), 1e-12)
    light = np.array([0.3, 0.8, 0.5])
    light /= np.linalg.norm(light)
    cam_c = -T[:3, :3].T @ T[:3, 3]
    for ti in np.argsort(-z[f].mean(1)):                       # far-to-near improves early z rejection
        a, b, c = f[ti]
        if min(z[a], z[b], z[c]) <= cam.near:
            continue
        p = uv[[a, b, c]]
        x0, x1 = max(math.floor(p[:, 0].min() - 0.5), 0), min(math.ceil(p[:, 0].max() - 0.5), W - 1)
        y0, y1 = max(math.floor(p[:, 1].min() - 0.5), 0), min(math.ceil(p[:, 1].max() - 0.5), H - 1)
        if x0 > x1 or y0 > y1:
            continue
        den = (p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1])
        if abs(den) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        l0 = ((p[1, 1] - p[2, 1]) * (xs - p[2, 0]) + (p[2, 0] - p[1, 0]) * (ys - p[2, 1])) / den
        l1 = ((p[2, 1] - p[0, 1]) * (xs - p[2, 0]) + (p[0, 0] - p[2, 0]) * (ys - p[2, 1])) / den
        l2 = 1 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        iz = l0 / z[a] + l1 / z[b] + l2 / z[c]                  # perspective-correct interpolation
        zz = 1.0 / iz
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        win = inside & (zz < sub)
        if not win.any():
            continue
        sub[win] = zz[win]
        w0, w1, w2 = (l0 / z[a]) * zz, (l1 / z[b]) * zz, (l2 / z[c]) * zz
        col = (w0[..., None] * vertex_colors[a] + w1[..., None] * vertex_colors[b] + w2[..., None] *
               vertex_colors[c])
        if shade:
            n = tri_n[ti] if np.dot(tri_n[ti], cam_c - v[a]) >= 0 else -tri_n[ti]   # two-sided
            col = col * (0.35 + 0.65 * max(float(np.dot(n, light)), 0.0))
        img[y0:y1 + 1, x0:x1 + 1][win] = col[win]
    mask = np.isfinite(zbuf)
    return img, mask, np.where(mask, zbuf, 0.0).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))
