"""E-T3 open-world evaluation helpers. Extends (never changes) the berkelium-text3d-eval/1 protocol in evaluate.py.

  align_and_score   B1: both meshes normalised to unit extent (largest side 1, bbox-centred); the prediction is
                    rotated by the best of the 24 axis-aligned rotations, then refined by rigid point-to-point ICP;
                    then evaluate.py's geometry_metrics. The SAME alignment is applied to every generator. It is
                    deliberately generous: it forgives any orientation/frame convention and absolute scale, so the
                    numbers measure shape, not pose.
  touch_components  connected components after merging pieces whose surfaces come within ``tol`` (0.02 normalized
                    units, the F-score threshold): evaluate.py's own rule "touching parts merge", made explicit for
                    multi-part assemblies whose parts are separate meshes. Applied identically to GT and prediction.
  view_embeddings   4 shaded views (geometry.render_mesh) -> evaluator CLIP image embeddings, averaged per mesh.
"""

from __future__ import annotations

import itertools

import numpy as np

from .geometry import geometry_metrics, sample_surface


def rotations_24() -> list[np.ndarray]:
    """The 24 proper rotations of the cube (signed permutation matrices with det +1)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs, strict=True)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def normalize_unit(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(0), v.max(0)
    return (v - (lo + hi) / 2) / max(float((hi - lo).max()), 1e-12)


def _chamfer(a, b, ta=None, tb=None):
    from scipy.spatial import cKDTree
    ta = ta or cKDTree(a)
    tb = tb or cKDTree(b)
    return 0.5 * (tb.query(a)[0].mean() + ta.query(b)[0].mean())


def icp(src: np.ndarray, dst: np.ndarray, iters: int = 30, tol: float = 1e-7):
    """Rigid point-to-point ICP (Kabsch, symmetric nearest neighbours). -> (R, t, chamfer)."""
    from scipy.spatial import cKDTree
    tree_d = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    prev = np.inf
    for _ in range(iters):
        cur = src @ R.T + t
        _, i = tree_d.query(cur)
        _, j = cKDTree(cur).query(dst)
        P = np.concatenate([src, src[j]])               # src->dst pairs and dst->src pairs
        Q = np.concatenate([dst[i], dst])
        mp, mq = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        t = mq - mp @ R.T
        err = _chamfer(src @ R.T + t, dst, tb=tree_d)
        if prev - err < tol:
            break
        prev = err
    return R, t, float(_chamfer(src @ R.T + t, dst, tb=tree_d))


def align_and_score(pred_v, pred_f, gt_v, gt_f, n_align: int = 4000, n_eval: int = 20000, seed: int = 0) -> dict:
    """B1 geometry: unit-extent normalisation, best of 24 rotations, ICP, then geometry_metrics (unit-extent units)."""
    if pred_v is None or pred_f is None or len(pred_f) == 0:
        return {**geometry_metrics(None, None, gt_v, gt_f), "alignment": None}
    pv, gv = normalize_unit(np.asarray(pred_v, np.float64)), normalize_unit(np.asarray(gt_v, np.float64))
    ps, _, _ = sample_surface(pv, pred_f, n_align, seed)
    gs, _, _ = sample_surface(gv, gt_f, n_align, seed + 1)
    from scipy.spatial import cKDTree
    tg = cKDTree(gs)
    coarse = sorted(((_chamfer(ps @ R.T, gs, tb=tg), k, R) for k, R in enumerate(rotations_24())), key=lambda x: x[0])
    best = None
    for c0, k, R0 in coarse[:3]:                        # refine the 3 best coarse candidates; keep the best result
        R1, t1, c1 = icp(ps @ R0.T, gs)
        if best is None or c1 < best[0]:
            best = (c1, k, R1 @ R0, t1, c0)
    c1, k, R, t, c0 = best
    out = geometry_metrics(pv @ R.T + t, pred_f, gv, gt_f, n=n_eval, seed=seed)
    out["alignment"] = {"units": "unit extent (largest bbox side = 1)", "coarse_rotation_index": k,
                        "coarse_chamfer": round(float(c0), 5), "icp_chamfer": round(float(c1), 5),
                        "R": np.round(R, 6).tolist(), "t": np.round(t, 6).tolist()}
    return out


def touch_components(v, f, tol: float = 0.02, n: int = 6000, seed: int = 0) -> int | None:
    """Connected components after merging mesh pieces whose sampled surfaces are within ``tol``."""
    if v is None or f is None or len(f) == 0:
        return None
    import trimesh
    from scipy.spatial import cKDTree
    pieces = trimesh.Trimesh(v, f, process=False).split(only_watertight=False)
    if len(pieces) == 1:
        return 1
    area = np.array([p.area for p in pieces])
    pts = [sample_surface(np.asarray(p.vertices), np.asarray(p.faces), max(50, int(n * a / area.sum())), seed + i)[0]
           for i, (p, a) in enumerate(zip(pieces, area, strict=True))]
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    trees = [cKDTree(p) for p in pts]
    for i, j in itertools.combinations(range(len(pieces)), 2):
        if trees[j].query(pts[i])[0].min() <= tol:
            parent[find(i)] = find(j)
    return len({find(i) for i in range(len(pieces))})


def view_images(v, f, colors, n_views: int = 4, res: int = 224, elevation: float = 25.0):
    """Shaded renders on white, the same cameras for every mesh (orbit radius 3.2, normalized_object units)."""
    from ..camera import orbit_cameras
    from .geometry import render_mesh
    cams = orbit_cameras(n_views, 3.2, elevation, res, res, phase_deg=30.0)
    if v is None:
        return [np.ones((res, res, 3), np.float32) for _ in cams]
    col = colors if colors is not None else np.full((len(v), 3), 0.7, np.float32)
    return [render_mesh(v, f, col, c)[0] for c in cams]


def r_precision(image_emb: np.ndarray, text_emb: np.ndarray, truth: list[int]) -> dict:
    """image_emb [N,D], text_emb [M,D] (L2-normalised), truth[i] = index of mesh i's own prompt.
    -> rank of the true prompt per mesh, R-precision@1 and @5."""
    sims = image_emb @ text_emb.T
    ranks = [int((sims[i] > sims[i, truth[i]]).sum()) + 1 for i in range(len(truth))]
    return {"ranks": ranks, "r_at_1": float(np.mean([r == 1 for r in ranks])),
            "r_at_5": float(np.mean([r <= 5 for r in ranks])), "n_candidates": int(text_emb.shape[0])}
