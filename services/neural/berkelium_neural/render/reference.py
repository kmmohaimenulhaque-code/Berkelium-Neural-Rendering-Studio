"""Berkelium CPU reference renderer (original code; PyTorch autograd; fp32 or fp64).

Two explicitly selected profiles (contracts.RenderProfile):
  gsplat_v1      reproduces the AMD gsplat 1.5.3 rules so its outputs/gradients are directly comparable
  berkelium_v02  GAUSSIAN_RENDERER_SPEC.md v0.2 rules (3-sigma support, cap 0.99, inclusive threshold)

The 2D stage (``rasterize``) consumes/produces gsplat's tensor layout (means2d, conics, colors, opacities,
isect_offsets, flatten_ids, render_alphas, last_ids) so it can serve as the oracle for both the gsplat backend
and the Berkelium HIP backward kernel.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..camera import T_cw
from ..contracts import CameraSpec, RenderConfig, RenderProfile
from .interface import GaussianTensors, RendererError, RenderOutput, covariance

ALPHA_THRESHOLD = 1.0 / 255.0
T_THRESHOLD = 1e-4


# ------------------------------------------------------------------------------------------------ project
def project(g: GaussianTensors, cam: CameraSpec, cfg: RenderConfig):
    """World Gaussians -> screen-space (means2d [N,2], conics [N,3], depths [N], radii [N,2] int32,
    opacities [N], colors [N,3]). Culled Gaussians have radii == 0. Differentiable w.r.t. all params."""
    means, q, scales, opac, colors = g.activated()
    dt, dev = means.dtype, means.device
    T = torch.tensor(T_cw(cam), dtype=dt, device=dev)
    R, t = T[:3, :3], T[:3, 3]
    mc = means @ R.T + t
    cov_c = R @ covariance(q, scales) @ R.T
    x, y, z = mc.unbind(-1)
    fx, fy, cx, cy, W, H = cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height
    gs = cfg.profile == RenderProfile.GSPLAT_V1
    zs = torch.where(z.abs() > 1e-12, z, torch.full_like(z, 1e-12))   # culled entries only; avoids inf in graph
    if gs:   # gsplat clamps the Jacobian evaluation point to 1.3x the field of view
        tfx, tfy = 0.5 * W / fx, 0.5 * H / fy
        tx = zs * torch.clamp(x / zs, min=-(cx / fx + 0.3 * tfx), max=(W - cx) / fx + 0.3 * tfx)
        ty = zs * torch.clamp(y / zs, min=-(cy / fy + 0.3 * tfy), max=(H - cy) / fy + 0.3 * tfy)
    else:
        tx, ty = x, y
    O = torch.zeros_like(z)
    J = torch.stack([fx / zs, O, -fx * tx / zs**2, O, fy / zs, -fy * ty / zs**2], -1).reshape(-1, 2, 3)
    cov2d = J @ cov_c @ J.transpose(-1, -2) + cfg.eps2d * torch.eye(2, dtype=dt, device=dev)
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    det_s = torch.where(det > 0, det, torch.ones_like(det))
    conics = torch.stack([cov2d[:, 1, 1] / det_s, -0.5 * (cov2d[:, 0, 1] + cov2d[:, 1, 0]) / det_s,
                          cov2d[:, 0, 0] / det_s], -1)
    means2d = torch.stack([fx * x / zs + cx, fy * y / zs + cy], -1)

    with torch.no_grad():
        if gs:
            valid = (z >= cam.near) & (z <= cam.far) & (det > 0) & (opac >= ALPHA_THRESHOLD)
            extend = torch.clamp(torch.sqrt(2.0 * torch.log(torch.clamp(opac, min=ALPHA_THRESHOLD) / ALPHA_THRESHOLD)),
                                 max=3.33)
            rx = torch.ceil(extend * torch.sqrt(cov2d[:, 0, 0].clamp(min=0)))
            ry = torch.ceil(extend * torch.sqrt(cov2d[:, 1, 1].clamp(min=0)))
            valid &= (rx > 0) | (ry > 0)
            valid &= ~((means2d[:, 0] + rx <= 0) | (means2d[:, 0] - rx >= W) | (means2d[:, 1] + ry <= 0)
                       | (means2d[:, 1] - ry >= H))
        else:
            valid = (z > cam.near) & (z < cam.far) & (det > 0)
            tr, dd = cov2d[:, 0, 0] + cov2d[:, 1, 1], det.clamp(min=0)
            lmax = 0.5 * tr + torch.sqrt((0.25 * tr * tr - dd).clamp(min=0))
            rx = ry = torch.ceil(3.0 * torch.sqrt(lmax))
            valid &= ~((means2d[:, 0] + rx <= 0) | (means2d[:, 0] - rx >= W) | (means2d[:, 1] + ry <= 0)
                       | (means2d[:, 1] - ry >= H))
        radii = torch.stack([rx, ry], -1).to(torch.int32) * valid[:, None]
    return means2d, conics, z, radii, opac, colors


# -------------------------------------------------------------------------------------------- intersect
@torch.no_grad()
def isect_tiles(means2d, radii, depths, tile_size: int, width: int, height: int, max_pairs: int):
    """gsplat tile assignment: tiles in [floor((m-r)/ts), ceil((m+r)/ts)) clamped. Pairs sorted by the total key
    (tile_id, depth float bits, gaussian index) -> (isect_offsets [th,tw] int32, flatten_ids [M] int32)."""
    tw, th = math.ceil(width / tile_size), math.ceil(height / tile_size)
    # host-side computation (works for CPU or GPU inputs); returned tensors are CPU, the caller moves them
    m, r = means2d.detach().cpu().to(torch.float32), radii.cpu().to(torch.float32)
    radii, depths = radii.cpu(), depths.detach().cpu()
    tmin = torch.floor((m - r) / tile_size).to(torch.int64)
    tmax = torch.ceil((m + r) / tile_size).to(torch.int64)
    tmin[:, 0].clamp_(0, tw), tmin[:, 1].clamp_(0, th)
    tmax[:, 0].clamp_(0, tw), tmax[:, 1].clamp_(0, th)
    vis = (radii > 0).all(-1)
    nx, ny = (tmax - tmin).clamp(min=0).unbind(-1)
    cnt = (nx * ny) * vis
    M = int(cnt.sum())
    if M > max_pairs:
        raise RendererError("RESOURCE_LIMIT", f"{M} Gaussian/tile pairs exceed max_pairs={max_pairs}")
    gid = torch.repeat_interleave(torch.arange(len(cnt)), cnt)
    first = torch.cumsum(cnt, 0) - cnt
    local = torch.arange(M) - first[gid]
    tx = tmin[gid, 0] + local % nx[gid].clamp(min=1)
    ty = tmin[gid, 1] + local // nx[gid].clamp(min=1)
    tile = ty * tw + tx
    dbits = torch.from_numpy(depths.detach().to(torch.float32).cpu().numpy().view(np.int32).astype(np.int64)
                             & 0xFFFFFFFF)[gid]
    order = np.lexsort((gid.numpy(), dbits.numpy(), tile.numpy()))       # stable total key
    order = torch.from_numpy(order)
    tile_sorted = tile[order]
    flatten_ids = gid[order].to(torch.int32)
    counts = torch.bincount(tile_sorted, minlength=tw * th)
    offsets = (torch.cumsum(counts, 0) - counts).to(torch.int32).reshape(th, tw)
    return offsets, flatten_ids, {"n_pairs": M, "max_pairs_per_tile": int(counts.max()) if M else 0,
                                  "tiles": (tw, th)}


# -------------------------------------------------------------------------------------------- rasterize
def rasterize(means2d, conics, colors, opacities, background, width: int, height: int, tile_size: int,
              isect_offsets, flatten_ids, profile: RenderProfile = RenderProfile.GSPLAT_V1):
    """Differentiable front-to-back compositing on the recorded branch.
    Returns (colors [H,W,C] incl. background, alphas [H,W], last_ids [H,W] int32 index into flatten_ids)."""
    C, dt, dev = colors.shape[-1], means2d.dtype, means2d.device
    th, tw = isect_offsets.shape
    M = flatten_ids.numel()
    last = torch.zeros(height, width, dtype=torch.int32)
    bg = torch.zeros(C, dtype=dt, device=dev) if background is None else background.to(dt)
    gs = profile == RenderProfile.GSPLAT_V1
    rows_c, rows_a = [], []
    offs = isect_offsets.flatten().tolist() + [M]
    for ty in range(th):
        row_c, row_a = [], []
        for tx in range(tw):
            tid = ty * tw + tx
            s, e = offs[tid], offs[tid + 1]
            y0, x0 = ty * tile_size, tx * tile_size
            hh, ww = min(tile_size, height - y0), min(tile_size, width - x0)
            if e <= s:
                row_c.append(bg.expand(hh, ww, C))
                row_a.append(torch.zeros(hh, ww, dtype=dt, device=dev))
                continue
            ids = flatten_ids[s:e].long()
            py, px = torch.meshgrid(torch.arange(y0, y0 + hh, dtype=dt) + 0.5,
                                    torch.arange(x0, x0 + ww, dtype=dt) + 0.5, indexing="ij")
            px, py = px.reshape(-1, 1), py.reshape(-1, 1)                     # [P,1]
            mu, cn, op = means2d[ids], conics[ids], opacities[ids]
            dx, dy = mu[:, 0][None] - px, mu[:, 1][None] - py                  # [P,K] (gsplat: xy - p)
            sigma = 0.5 * (cn[:, 0] * dx * dx + cn[:, 2] * dy * dy) + cn[:, 1] * dx * dy
            G = torch.exp(-sigma)
            if gs:
                alpha = torch.clamp(op[None] * G, max=0.999)
                valid = (sigma >= 0) & (alpha >= ALPHA_THRESHOLD)
            else:
                alpha = torch.clamp(op[None] * G, max=0.99)
                valid = (2.0 * sigma) <= 9.0
            with torch.no_grad():
                a_v = torch.where(valid, alpha, torch.zeros_like(alpha))
                one_m = 1.0 - a_v
                T_before = torch.cumprod(torch.cat([torch.ones_like(one_m[:, :1]), one_m[:, :-1]], 1), 1)
                if gs:   # exclusive stop: the contribution that would bring T to <= 1e-4 is NOT applied
                    stop = valid & (T_before * (1.0 - a_v) <= T_THRESHOLD)
                    include = valid & (torch.cumsum(stop.to(torch.int32), 1) == 0)
                else:    # spec: stop BEFORE processing i once T < 1e-4; crossing contribution included
                    include = valid & (T_before >= T_THRESHOLD)
                kk = torch.arange(len(ids))[None].expand_as(include)
                lastk = torch.where(include, kk, torch.full_like(kk, -1)).max(1).values
                last_glob = torch.where(lastk >= 0, s + lastk, torch.zeros_like(lastk))
            a_inc = torch.where(include, alpha, torch.zeros_like(alpha))
            Tb = torch.cumprod(torch.cat([torch.ones_like(a_inc[:, :1]), (1.0 - a_inc)[:, :-1]], 1), 1)
            w = Tb * a_inc                                                      # [P,K]
            T_final = torch.prod(1.0 - a_inc, 1)                                # [P]
            col = w @ colors[ids] + T_final[:, None] * bg[None]
            row_c.append(col.reshape(hh, ww, C))
            row_a.append((1.0 - T_final).reshape(hh, ww))
            last[y0:y0 + hh, x0:x0 + ww] = last_glob.reshape(hh, ww).to(torch.int32)
        rows_c.append(torch.cat(row_c, 1))
        rows_a.append(torch.cat(row_a, 1))
    out_c = torch.cat(rows_c, 0)
    out_a = torch.cat(rows_a, 0)
    return out_c, out_a, last


# -------------------------------------------------------------------------------------------- renderer
class ReferenceRenderer:
    """CPU (or any torch device) reference. Deterministic, autograd-differentiable, fp32/fp64."""

    name = "reference_cpu"

    def capabilities(self) -> dict:
        return {"backend": "reference", "profiles": [p.value for p in RenderProfile], "outputs": ["rgb", "alpha",
                "depth"], "gradients": "autograd", "deterministic": True, "camera_gradients": False,
                "max_recommended_pixels": 256 * 256}

    def prepare(self, cfg: RenderConfig) -> str:
        return "cpu"

    def render(self, g: GaussianTensors, cam: CameraSpec, cfg: RenderConfig) -> RenderOutput:
        if g.means.shape[0] > cfg.max_gaussians:
            raise RendererError("RESOURCE_LIMIT", f"{g.means.shape[0]} Gaussians > max_gaussians")
        for p in g.params():
            if not torch.isfinite(p).all():
                raise RendererError("INVALID_SCENE", "non-finite Gaussian parameters")
        means2d, conics, depths, radii, opac, colors = project(g, cam, cfg)
        offsets, flat, info = isect_tiles(means2d, radii, depths, cfg.tile_size, cam.width, cam.height,
                                          cfg.max_pairs)
        feats = torch.cat([colors, depths[:, None]], -1)                   # depth rides as channel 4 (gsplat ED)
        bg = torch.tensor([*cfg.background, 0.0], dtype=means2d.dtype)
        img, alpha, last = rasterize(means2d, conics, feats, opac, bg, cam.width, cam.height, cfg.tile_size,
                                     offsets, flat, cfg.profile)
        rgb, dacc = img[..., :3], img[..., 3]
        if cfg.profile == RenderProfile.GSPLAT_V1:
            depth = dacc / alpha.clamp(min=1e-10)                           # gsplat "ED"
        else:
            depth = torch.where(alpha >= 1e-6, dacc / alpha.clamp(min=1e-6), torch.zeros_like(dacc))
        meta = {"backend": self.name, "device": str(means2d.device), "dtype": str(means2d.dtype),
                "profile": cfg.profile.value, "deterministic": True, "n_gaussians": int(g.means.shape[0]),
                "n_visible": int((radii > 0).all(-1).sum()), **info}
        aux = {"means2d": means2d, "conics": conics, "colors": feats, "opacities": opac, "depths": depths,
               "radii": radii, "isect_offsets": offsets, "flatten_ids": flat, "last_ids": last,
               "render_alphas": alpha, "background": bg}
        return RenderOutput(rgb=rgb, alpha=alpha, depth=depth, meta=meta, aux=aux)
