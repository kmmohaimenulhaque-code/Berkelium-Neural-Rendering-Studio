"""CPU-only diagnosis of the forward discrepancy between an FP32 renderer and the FP64 oracle on the E-R1 cases.

Background: on MI300X (E-R1, 2026-10-13) gsplat and the Berkelium backend (which reuses gsplat's forward) both
reported forward RGB max-abs error 1.0456e-3 vs the FP64 CPU oracle on the 100k-Gaussian case, while alpha stayed at
3.2e-5. This script does NOT reproduce the GPU numbers; it runs the Berkelium reference renderer's own rules (profile
gsplat_v1) on the same scenes/cameras and splits FP32-vs-FP64 forward error into its sources:

  A  oracle            FP64 projection + FP64 compositing (what E-R1 compared against)
  B  fp32_projection   FP32 projection (means2d, conics, depths, radii, opacities, colours) -> cast to FP64 ->
                       FP64 compositing                      (isolates projection rounding incl. its discrete effects)
  C  fp32_all          FP32 projection + FP32 compositing    (an FP32 implementation of the same rules, like gsplat)
  D  fp32_pair_list    FP64 projection values, but the Gaussian/tile pair list (tile membership + per-tile depth
                       order) built from the FP32 projection; FP64 compositing   (isolates the discrete pair list)

If D carries most of C's RGB error with near-zero alpha error, the discrepancy is driven by the per-tile depth order
of near-equal-depth Gaussians (alpha is order-invariant, colour is not) rather than by arithmetic rounding.

  python -m benchmarks.renderer.forward_precision_diagnosis --out results/renderer/E-R1_forward_diagnosis_cpu.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from berkelium_neural.contracts import RenderConfig
from berkelium_neural.render.interface import GaussianTensors
from berkelium_neural.render.reference import ALPHA_THRESHOLD, T_THRESHOLD, isect_tiles, project, rasterize

from benchmarks.common import ROOT, load_protocol, run_metadata
from benchmarks.renderer.bench_renderer import case_camera, synthetic_scene

TOL = 2e-4          # the spec forward tolerance used by the E-R1 gate cases (not changed here)


def _composite(m2, con, col, z, op, offs, flat, cam, cfg, dt):
    feats = torch.cat([col, z[:, None]], -1).to(dt)
    bg = torch.tensor([*cfg.background, 0.0], dtype=dt)
    img, alpha, _ = rasterize(m2.to(dt), con.to(dt), feats, op.to(dt), bg, cam.width, cam.height, cfg.tile_size,
                              offs, flat, cfg.profile)
    return img[..., :3].double(), alpha.double()


def _err(rgb, alpha, ref_rgb, ref_alpha) -> dict:
    er, ea = (rgb - ref_rgb).abs(), (alpha - ref_alpha).abs()
    px = er.amax(-1)
    i = int(px.argmax())
    y, x = divmod(i, px.shape[1])
    return {"rgb_max_abs_err": float(er.max()), "alpha_max_abs_err": float(ea.max()),
            "rgb_pixels_over_tol": int((px > TOL).sum()), "alpha_pixels_over_tol": int((ea > TOL).sum()),
            "pixels": int(px.numel()), "at_worst_rgb_pixel": {"y": y, "x": x, "rgb_err": float(px[y, x]),
                                                             "alpha_err": float(ea[y, x]),
                                                             "oracle_alpha": float(ref_alpha[y, x])}}


def _pair_list_order_differs(offs_a, flat_a, offs_b, flat_b) -> dict:
    same_tiles = bool(torch.equal(offs_a, offs_b)) and flat_a.numel() == flat_b.numel()
    return {"pairs_oracle": int(flat_a.numel()), "pairs_fp32": int(flat_b.numel()),
            "identical_pair_list": bool(same_tiles and torch.equal(flat_a, flat_b)),
            "same_tile_membership_counts": same_tiles,
            "pairs_at_different_position": int((flat_a != flat_b).sum()) if same_tiles else None}


def _pixel_trace(m2, con, op, offs, flat, y, x, ts):
    """Per-Gaussian compositing decisions at one pixel (same rules as reference.rasterize, gsplat_v1), FP64."""
    tw = offs.shape[1]
    tid = (y // ts) * tw + (x // ts)
    o = offs.flatten().tolist() + [flat.numel()]
    ids = flat[o[tid]:o[tid + 1]].long()
    mu, cn, opa = m2[ids].double(), con[ids].double(), op[ids].double()
    dx, dy = mu[:, 0] - (x + 0.5), mu[:, 1] - (y + 0.5)
    sigma = 0.5 * (cn[:, 0] * dx * dx + cn[:, 2] * dy * dy) + cn[:, 1] * dx * dy
    alpha = torch.clamp(opa * torch.exp(-sigma), max=0.999)
    valid = (sigma >= 0) & (alpha >= ALPHA_THRESHOLD)
    a_v = torch.where(valid, alpha, torch.zeros_like(alpha))
    T_before = torch.cumprod(torch.cat([a_v.new_ones(1), (1 - a_v)[:-1]]), 0)
    stop = valid & (T_before * (1 - a_v) <= T_THRESHOLD)
    include = valid & (torch.cumsum(stop.to(torch.int32), 0) == 0)
    return {int(g): (bool(inc), float(a), float(t), bool(sg >= 0), bool(st))
            for g, inc, a, t, sg, st in zip(ids, include, alpha, T_before, sigma, stop, strict=True)}


def _decision_flips(p64, p32, offs_a, flat_a, offs_b, flat_b, y, x, ts) -> list[dict]:
    A = _pixel_trace(p64[0], p64[1], p64[4], offs_a, flat_a, y, x, ts)
    B = _pixel_trace(p32[0], p32[1], p32[4], offs_b, flat_b, y, x, ts)
    cols = p64[5]
    flips = []
    for g in sorted(set(A) | set(B)):
        a, b = A.get(g), B.get(g)
        if a is None or b is None or a[0] != b[0]:
            ref = a or b
            cause = ("tile membership" if a is None or b is None else
                     "alpha >= 1/255 threshold" if (a[1] >= ALPHA_THRESHOLD) != (b[1] >= ALPHA_THRESHOLD) else
                     "sigma >= 0" if a[3] != b[3] else "T <= 1e-4 termination" if a[4] != b[4] else "upstream flip")
            flips.append({"gaussian": g, "included_oracle": a[0] if a else None, "included_fp32": b[0] if b else None,
                          "alpha_oracle_fp64": ref[1] if a else None, "alpha_fp32_projection": b[1] if b else None,
                          "alpha_threshold": ALPHA_THRESHOLD, "T_before_oracle": a[2] if a else None,
                          "colour": [float(c) for c in cols[g]], "cause": cause,
                          "approx_rgb_effect_bound": (ref[1] * (a[2] if a else b[2]))})
    return flips


@torch.no_grad()
def run_case(case: dict) -> dict:
    t0 = time.perf_counter()
    s = synthetic_scene(case["n"], case["seed"])
    cam = case_camera(case)
    cfg = RenderConfig()
    p64 = project(GaussianTensors.from_scene(s, dtype=torch.float64), cam, cfg)
    p32 = project(GaussianTensors.from_scene(s, dtype=torch.float32), cam, cfg)
    m2a, cona, za, ra, opa, cola = p64
    m2b, conb, zb, rb, opb, colb = p32
    offs_a, flat_a, info = isect_tiles(m2a, ra, za, cfg.tile_size, cam.width, cam.height, 10**9)
    offs_b, flat_b, _ = isect_tiles(m2b, rb, zb, cfg.tile_size, cam.width, cam.height, 10**9)
    A = _composite(m2a, cona, cola, za, opa, offs_a, flat_a, cam, cfg, torch.float64)
    B = _composite(m2b, conb, colb, zb, opb, offs_b, flat_b, cam, cfg, torch.float64)
    C = _composite(m2b, conb, colb, zb, opb, offs_b, flat_b, cam, cfg, torch.float32)
    D = _composite(m2a, cona, cola, za, opa, offs_b, flat_b, cam, cfg, torch.float64)
    errs = {"B_fp32_projection": _err(*B, *A), "C_fp32_all": _err(*C, *A), "D_fp32_pair_list_only": _err(*D, *A)}
    w = errs["C_fp32_all"]["at_worst_rgb_pixel"]
    flips = _decision_flips(p64, p32, offs_a, flat_a, offs_b, flat_b, w["y"], w["x"], cfg.tile_size)
    return {"case": case, "workload": info, "pair_list": _pair_list_order_differs(offs_a, flat_a, offs_b, flat_b),
            "vs_oracle": errs, "worst_pixel_decision_flips_fp64_vs_fp32_projection": flips,
            "wall_s": round(time.perf_counter() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="benchmarks/configs/renderer_mi300x_v1.json")
    ap.add_argument("--max-n", type=int, default=10**9, help="skip cases with more Gaussians (CPU time)")
    ap.add_argument("--out", default="results/renderer/E-R1_forward_diagnosis_cpu.json")
    a = ap.parse_args(argv)
    proto = load_protocol(a.protocol)
    cases = [c for c in proto["cases"] if c["n"] <= a.max_n]
    res = {"diagnosis": "FP32 vs FP64 forward error decomposition (reference renderer rules, CPU)",
           "scope": "CPU reference renderer only; NOT a GPU measurement and NOT a rerun of E-R1",
           "tolerance_shown": TOL, **run_metadata(vars(a), "benchmarks/renderer/forward_precision_diagnosis.py"),
           "cases": []}
    for c in cases:
        r = run_case(c)
        res["cases"].append(r)
        print(json.dumps({"case": c, **{k: {kk: v[kk] for kk in ("rgb_max_abs_err", "alpha_max_abs_err")}
                                         for k, v in r["vs_oracle"].items()}, "wall_s": r["wall_s"]}))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(out)


if __name__ == "__main__":
    main()
