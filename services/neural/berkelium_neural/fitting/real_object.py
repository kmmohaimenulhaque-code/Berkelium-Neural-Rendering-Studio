"""M7: fit a Gaussian scene to multi-view observations of a real object, evaluate on held-out views.

Data: Google Scanned Objects (GSO) — photogrammetry scans of real household objects by Google Research, CC-BY 4.0,
fetched from Gazebo Fuel at a pinned version and checked against a recorded sha256. Observations are images of the
textured scan rendered from known cameras (the standard way GSO is used for evaluation); they are NOT raw
photographs, and camera poses are exact.

What is fitted and what is not:
  * The fit sees only the training observations (RGB + silhouette) and their cameras.
  * Initialisation is space carving from the training silhouettes (no mesh, no depth, no scan points).
  * The scan mesh is used only (a) to render the observations and (b) for the geometric evaluation afterwards.
  * Fixed Gaussian count (no densification/pruning), Berkelium reference renderer, profile gsplat_v1.
Evaluation: PSNR / SSIM / silhouette IoU on held-out cameras never used for fitting, before and after the fit,
and Chamfer / F-score of opaque Gaussian centres against the scan surface.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..camera import T_cw, orbit_cameras
from ..contracts import CameraSpec

GSO_BASE = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
# Pinned objects: name -> sha256 of the Fuel zip (version 1). A changed download is refused, never used silently.
GSO_PINS = {
    "Schleich_Lion_Action_Figure": "ae402d06517c83a4eea39f4e7351be13111e36759a9266b0fbb91b6843622ad9",
    "Racoon": "1408b68f9eb7fd5f34c88130735331a761deb02220320cf82f51804f2c1f5dd7",
    "Room_Essentials_Bowl_Turquiose": "6eb64260335468ecaf21d8b6d5b20fedbaf1b75800af2b33121ceccb2462b4c9",
}
GSO_LICENSE = "CC-BY 4.0 (Google LLC, Google Scanned Objects; attribution in THIRD_PARTY_NOTICES.md)"
EXTENT = 1.6        # scans are normalised so their largest side is 1.6 (normalized_object units)


# ------------------------------------------------------------------------------------------------- data
def fetch_gso(name: str, root: Path) -> Path:
    """Download (once) and verify a pinned GSO object. Returns the zip path."""
    if name not in GSO_PINS:
        raise ValueError(f"{name!r} is not a pinned GSO object ({sorted(GSO_PINS)})")
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    z = d / f"{name}.zip"
    if not z.exists():
        import urllib.request
        tmp = z.with_suffix(".part")
        urllib.request.urlretrieve(f"{GSO_BASE}/{name}/1/{name}.zip", tmp)
        tmp.rename(z)
    sha = hashlib.sha256(z.read_bytes()).hexdigest()
    if sha != GSO_PINS[name]:
        raise ValueError(f"{z} sha256 {sha} != pinned {GSO_PINS[name]}; refusing to use a changed download")
    return z


@dataclass
class ScanMesh:
    v: np.ndarray            # [V,3] normalised, +Y up
    f: np.ndarray            # [F,3]
    uv: np.ndarray           # [V,2]
    tex: np.ndarray          # [T,T,3] float32 in [0,1]
    transform: dict          # how the scan was normalised (for the record)
    source: dict


def load_scan(zip_path: Path, tex_size: int = 1024) -> ScanMesh:
    import trimesh
    from PIL import Image
    z = zipfile.ZipFile(zip_path)
    m = trimesh.load(io.BytesIO(z.read("meshes/model.obj")), file_type="obj", process=False, force="mesh")
    v = np.asarray(m.vertices, np.float64)
    v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)                  # GSO is Z-up -> Berkelium +Y up
    lo, hi = v.min(0), v.max(0)
    center, scale = (lo + hi) / 2, EXTENT / float((hi - lo).max())
    v = (v - center) * scale
    tex = Image.open(io.BytesIO(z.read("materials/textures/texture.png"))).convert("RGB")
    tex = np.asarray(tex.resize((tex_size, tex_size), Image.Resampling.BOX), np.float32) / 255.0
    return ScanMesh(v=v, f=np.asarray(m.faces, np.int64), uv=np.asarray(m.visual.uv, np.float64), tex=tex,
                    transform={"axes": "GSO z-up -> y-up (x, z, -y)", "center_m": center.tolist(),
                               "scale_per_m": scale, "extent": EXTENT},
                    source={"zip": zip_path.name, "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                            "license": GSO_LICENSE, "vertices": int(len(v)), "faces": int(len(m.faces))})


def _sample_tex(tex, uv):
    """Bilinear texture lookup; uv in [0,1], v up (OBJ convention)."""
    T = tex.shape[0]
    x = np.clip(uv[..., 0] % 1.0, 0, 1) * (T - 1)
    y = np.clip(1.0 - (uv[..., 1] % 1.0), 0, 1) * (T - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, T - 1), np.minimum(y0 + 1, T - 1)
    wx, wy = (x - x0)[..., None], (y - y0)[..., None]
    return ((tex[y0, x0] * (1 - wx) + tex[y0, x1] * wx) * (1 - wy) + (tex[y1, x0] * (1 - wx) + tex[y1, x1] * wx) * wy)


def render_scan(scan: ScanMesh, cam: CameraSpec, ss: int = 2, with_depth: bool = False):
    """Unlit albedo render of the textured scan (z-buffer, perspective-correct UVs, ss x ss supersampling).
    -> (rgb premultiplied on black [H,W,3], alpha [H,W]) at the camera's resolution; with_depth adds the mean
    camera-Z of the covered subsamples [H,W] (0 where empty)."""
    H, W = cam.height * ss, cam.width * ss
    T = T_cw(cam)
    pc = scan.v @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    fx, fy, cx, cy = cam.fx * ss, cam.fy * ss, cam.cx * ss, cam.cy * ss
    with np.errstate(divide="ignore", invalid="ignore"):
        px = np.stack([fx * pc[:, 0] / z + cx, fy * pc[:, 1] / z + cy], 1)
    zbuf = np.full((H, W), np.inf)
    ubuf = np.zeros((H, W, 2))
    f = scan.f
    for ti in np.argsort(z[f].mean(1)):                               # near-to-far: early z rejection
        a, b, c = f[ti]
        if min(z[a], z[b], z[c]) <= cam.near:
            continue
        p = px[[a, b, c]]
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
        w0, w1, w2 = l0 / z[a], l1 / z[b], l2 / z[c]
        iz = w0 + w1 + w2
        zz = 1.0 / iz
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        upd = inside & (zz < sub)
        if not upd.any():
            continue
        sub[upd] = zz[upd]
        uvp = (w0[..., None] * scan.uv[a] + w1[..., None] * scan.uv[b] + w2[..., None] * scan.uv[c]) / iz[..., None]
        ubuf[y0:y1 + 1, x0:x1 + 1][upd] = uvp[upd]
    hit = np.isfinite(zbuf)
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[hit] = _sample_tex(scan.tex, ubuf[hit])
    a = hit.astype(np.float32)
    rgb = rgb.reshape(cam.height, ss, cam.width, ss, 3).mean((1, 3))
    a_full = a
    a = a.reshape(cam.height, ss, cam.width, ss).mean((1, 3))
    if not with_depth:
        return rgb.astype(np.float32), a.astype(np.float32)
    zsum = np.where(hit, zbuf, 0.0).reshape(cam.height, ss, cam.width, ss).sum((1, 3))
    depth = zsum / np.maximum(a_full.reshape(cam.height, ss, cam.width, ss).sum((1, 3)), 1)
    return rgb.astype(np.float32), a.astype(np.float32), depth.astype(np.float32)


def view_split(res: int, n_train: int = 24, n_test: int = 8, radius: float = 3.0, fov: float = 40.0):
    """Training cameras on two rings (elevation 15 and 40 deg); held-out cameras on a third elevation (28 deg)
    at azimuths offset from every training camera."""
    half = n_train // 2
    train = (orbit_cameras(half, radius, 15.0, res, res, fov) +
             orbit_cameras(n_train - half, radius, 40.0, res, res, fov, phase_deg=180.0 / (n_train - half)))
    test = orbit_cameras(n_test, radius, 28.0, res, res, fov, phase_deg=360.0 / n_test / 2 + 7.0)
    return train, test


# ----------------------------------------------------------------------------------------------- init
def space_carve(cams, alphas, n: int, seed: int = 0, grid: int = 64, bound: float = 1.0, thresh: float = 0.5):
    """Visual hull from silhouettes only: keep voxels whose centre projects inside every training silhouette,
    then sample ``n`` points on the hull's surface voxels. -> (points [n,3], voxel_size, occupied_count)."""
    from ..camera import project_points
    ax = (np.arange(grid) + 0.5) / grid * 2 * bound - bound
    X = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
    occ = np.ones(len(X), bool)
    for cam, a in zip(cams, alphas, strict=True):
        uv, z = project_points(cam, X)
        ui, vi = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        ok = (z > 0) & (ui >= 0) & (vi >= 0) & (ui < cam.width) & (vi < cam.height)
        inside = np.zeros(len(X), bool)
        inside[ok] = a[vi[ok], ui[ok]] >= thresh
        occ &= inside
    vol = occ.reshape(grid, grid, grid)
    pad = np.pad(vol, 1)
    interior = vol.copy()
    for d in range(3):
        for s in (-1, 1):
            interior &= np.roll(pad, s, axis=d)[1:-1, 1:-1, 1:-1]
    surf = (vol & ~interior).reshape(-1)
    pts = X[surf]
    if len(pts) == 0:
        raise ValueError("space carving removed everything: silhouettes and cameras disagree")
    rng = np.random.default_rng(seed)
    vs = 2 * bound / grid
    pick = rng.choice(len(pts), size=n, replace=len(pts) < n)
    return pts[pick] + rng.uniform(-0.5, 0.5, (n, 3)) * vs, vs, int(vol.sum())


def init_colors(points, cams, rgbs, alphas):
    """Mean observed colour at each point's projection over the views whose silhouette contains it."""
    from ..camera import project_points
    acc, cnt = np.zeros((len(points), 3)), np.zeros(len(points))
    for cam, rgb, a in zip(cams, rgbs, alphas, strict=True):
        uv, z = project_points(cam, points)
        ui, vi = np.clip(uv[:, 0].astype(int), 0, cam.width - 1), np.clip(uv[:, 1].astype(int), 0, cam.height - 1)
        w = ((z > 0) & (a[vi, ui] > 0.5)).astype(float)
        acc += w[:, None] * rgb[vi, ui] / np.maximum(a[vi, ui], 1e-3)[:, None]
        cnt += w
    return np.clip(acc / np.maximum(cnt, 1)[:, None], 0.02, 0.98)


# ------------------------------------------------------------------------------------------------ fit
def fit(train_cams, train_rgb, train_a, n: int, steps: int, seed: int = 0, log=print, checkpoint: Path | None = None):
    """Fit a fixed-count Gaussian scene to the training observations with the Berkelium reference renderer."""
    import torch

    from ..contracts import RenderConfig
    from ..render.interface import GaussianTensors
    from ..render.reference import ReferenceRenderer
    from ..scene import GaussianScene
    torch.manual_seed(seed)
    pts, vs, n_occ = space_carve(train_cams, train_a, n, seed)
    col = init_colors(pts, train_cams, train_rgb, train_a)
    init = GaussianScene(pts, np.full((n, 3), math.log(0.6 * vs)), np.tile([1.0, 0, 0, 0], (n, 1)),
                         np.full(n, math.log(0.3 / 0.7)), np.log(col / (1 - col)),
                         meta={"init": "space_carving(train silhouettes)", "voxel": vs, "occupied_voxels": n_occ})
    g = GaussianTensors.from_scene(init, requires_grad=True)
    opt = torch.optim.Adam([{"params": [g.means], "lr": 2e-3}, {"params": [g.log_scales], "lr": 5e-3},
                            {"params": [g.quats], "lr": 2e-3}, {"params": [g.opacity_logits], "lr": 3e-2},
                            {"params": [g.color_logits], "lr": 2e-2}])
    R, cfg = ReferenceRenderer(), RenderConfig(background=(0.0, 0.0, 0.0))
    targets = [(torch.from_numpy(c), torch.from_numpy(a)) for c, a in zip(train_rgb, train_a, strict=True)]
    order = np.random.default_rng(seed + 1).permutation(np.arange(steps) % len(train_cams))
    start, hist, t0 = 0, [], time.time()
    if checkpoint is not None and Path(checkpoint).exists():
        st = torch.load(checkpoint, weights_only=False)
        for p, q in zip(g.params(), st["params"], strict=True):
            p.data.copy_(q)
        opt.load_state_dict(st["opt"])
        start, hist = st["step"], st["hist"]
    for it in range(start, steps):
        i = int(order[it])
        opt.zero_grad(set_to_none=True)
        o = R.render(g, train_cams[i], cfg)
        tc, ta = targets[i]
        loss = (o.rgb - tc).abs().mean() + 0.5 * (o.alpha - ta).abs().mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            g.log_scales.clamp_(math.log(1e-3), math.log(0.3))
        if (it + 1) % 50 == 0 or it + 1 == steps:
            hist.append({"step": it + 1, "loss": round(loss.item(), 5), "s": round(time.time() - t0, 1)})
            log(f"  step {it + 1}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
            if checkpoint is not None:
                torch.save({"params": [p.detach().clone() for p in g.params()], "opt": opt.state_dict(),
                            "step": it + 1, "hist": hist}, checkpoint)
    fitted = GaussianScene(*(p.detach().numpy().astype(np.float64) for p in g.params()),
                           meta={**init.meta, "fit_steps": steps, "renderer": R.name, "profile": cfg.profile.value})
    return init, fitted, hist


# ------------------------------------------------------------------------------------------- evaluate
def render_scene(scene, cams, with_depth: bool = False):
    import torch

    from ..contracts import RenderConfig
    from ..render.interface import GaussianTensors
    from ..render.reference import ReferenceRenderer
    R, cfg = ReferenceRenderer(), RenderConfig(background=(0.0, 0.0, 0.0))
    g = GaussianTensors.from_scene(scene)
    with torch.no_grad():
        outs = [R.render(g, c, cfg) for c in cams]
    if with_depth:
        return [o.rgb.numpy() for o in outs], [o.alpha.numpy() for o in outs], [o.depth.numpy() for o in outs]
    return [o.rgb.numpy() for o in outs], [o.alpha.numpy() for o in outs]


def depth_metrics(pred_d, pred_a, gt_d, gt_a) -> dict:
    """Rendered expected depth vs the scan's depth on held-out views, over pixels fully covered by the scan and
    opaque (alpha > 0.5) in the prediction. Measures the surface the fit renders, which centre distances do not."""
    errs, cov = [], []
    for pd, pa, gd, ga in zip(pred_d, pred_a, gt_d, gt_a, strict=True):
        full = ga >= 0.999
        m = full & (pa > 0.5)
        cov.append(float(m.sum() / max(full.sum(), 1)))
        errs.append(np.abs(pd[m] - gd[m]))
    e = np.concatenate(errs) if errs else np.zeros(0)
    if e.size == 0:
        return {"pixels": 0, "mean_abs": None}
    return {"pixels": int(e.size), "coverage": float(np.mean(cov)), "mean_abs": float(e.mean()),
            "median_abs": float(np.median(e)), "within_0.02": float((e < 0.02).mean()),
            "within_0.05": float((e < 0.05).mean()), "units": "camera Z, normalised object units (extent 1.6)"}


def image_metrics(pred_rgb, pred_a, gt_rgb, gt_a) -> dict:
    from skimage.metrics import structural_similarity
    ps, ss, iou = [], [], []
    for pr, pa, gr, ga in zip(pred_rgb, pred_a, gt_rgb, gt_a, strict=True):
        mse = float(((pr - gr) ** 2).mean())
        ps.append(10 * math.log10(1.0 / max(mse, 1e-12)))
        ss.append(float(structural_similarity(pr, gr, channel_axis=-1, data_range=1.0)))
        pm, gm = pa > 0.5, ga > 0.5
        iou.append(float((pm & gm).sum() / max((pm | gm).sum(), 1)))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "mask_iou": float(np.mean(iou)),
            "psnr_per_view": [round(x, 3) for x in ps]}


def geometry_metrics(scene, scan: ScanMesh, opacity_min: float = 0.25, n_surface: int = 20000) -> dict:
    """Chamfer-L1 / F-score between Gaussian centres with opacity >= opacity_min and points sampled on the scan surface
    (evaluation only). 0.25 keeps every initial Gaussian (initial opacity 0.3), so init and fit are compared alike."""
    from scipy.spatial import cKDTree

    from ..text3d.geometry import sample_surface
    op = 1 / (1 + np.exp(-scene.opacity_logits))
    c = scene.means[op >= opacity_min]
    if len(c) == 0:
        return {"n_opaque": 0, "opacity_min": opacity_min, "chamfer_l1": None, "fscore@0.02": None, "fscore@0.05": None}
    s, _, _ = sample_surface(scan.v, scan.f, n_surface, 0)
    d1, _ = cKDTree(s).query(c)
    d2, _ = cKDTree(c).query(s)
    f = {}
    for t in (0.02, 0.05):
        p, r = float((d1 < t).mean()), float((d2 < t).mean())
        f[f"fscore@{t}"] = 2 * p * r / max(p + r, 1e-12)
    return {"n_opaque": int(len(c)), "opacity_min": opacity_min, "chamfer_l1": float(0.5 * (d1.mean() + d2.mean())),
            **f}


def contact_sheet(rows: list[list[np.ndarray]]) -> np.ndarray:
    return np.concatenate([np.concatenate(r, 1) for r in rows], 0)


def save_png(path: Path, img: np.ndarray):
    from PIL import Image
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(path)


def write_json(path: Path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
