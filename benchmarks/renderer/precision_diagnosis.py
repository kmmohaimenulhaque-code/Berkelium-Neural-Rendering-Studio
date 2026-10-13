"""Precision diagnosis for the division-based transmittance recovery used by gsplat's backward (and therefore by the
Berkelium kernel, which deliberately reproduces gsplat's per-pixel math).

Builds an FP64 copy of csrc/bk_det_bwd.hip (mechanical float->double rewrite, scratch only) under HIP-CPU and compares
2D gradients (means2d, conics, colours, opacities) on one scene against the FP64 autograd oracle:
  - kernel FP32 (division-based T recovery)    -> rounding error of the algorithm in FP32
  - autograd FP32 (cumprod forward)             -> FP32 baseline of a different (product-form) algorithm
  - kernel FP64 (same algorithm)                -> if ~1e-12: the kernel logic is exact; error is precision only

  python -m benchmarks.renderer.precision_diagnosis --n 3000 --res 128 --out results/renderer/E-R0_precision.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from berkelium_neural import camera as cm
from berkelium_neural.contracts import RenderConfig
from berkelium_neural.render.berkelium_ext import (
    CSRC,
    hip_cpu_include,
    rasterize_bwd_hipcpu,
    segments,
)
from berkelium_neural.render.interface import GaussianTensors
from berkelium_neural.render.reference import isect_tiles, project, rasterize

from benchmarks.common import ROOT, run_metadata
from benchmarks.renderer.bench_renderer import synthetic_scene

NAMES = ("v_means2d", "v_conics", "v_colors", "v_opacities")


def build_fp64(tmp: Path) -> Path:
    for f in ("bk_det_bwd.h", "bk_det_bwd.hip", "hipcpu_capi.cpp"):
        s = (CSRC / f).read_text()
        s = re.sub(r"\bfloat\b", "double", s)
        s = s.replace("expf(x)", "exp(x)").replace("fminf(", "fmin(").replace("fabsf(", "fabs(")
        s = s.replace("0.999f", "0.999").replace("1.f / 255.f", "1.0 / 255.0")
        (tmp / f).write_text(s)
    lib = tmp / "libbk64.so"
    subprocess.run(["g++", "-std=c++17", "-O2", "-fPIC", "-shared", f"-I{tmp}", f"-I{hip_cpu_include()}", "-x", "c++",
                    str(tmp / "bk_det_bwd.hip"), "-x", "c++", str(tmp / "hipcpu_capi.cpp"), "-o", str(lib), "-ltbb",
                    "-lpthread"], check=True, capture_output=True)
    return lib


def run(n: int, res: int, seed: int) -> dict:
    s = synthetic_scene(n, seed)
    cam = cm.make_camera((0.2, -0.15, -0.6), (0, 0, 3), res, res, fov_y_deg=55)
    gen = torch.Generator().manual_seed(seed)
    vrc, vra = torch.randn(res, res, 4, generator=gen), torch.randn(res, res, generator=gen)
    out = {}
    for dt in (torch.float64, torch.float32):
        g = GaussianTensors.from_scene(s, dtype=dt)
        with torch.no_grad():
            m2, con, z, radii, op, col = project(g, cam, RenderConfig())
        offs, flat, info = isect_tiles(m2, radii, z, 16, res, res, 10**9)
        L = [t.detach().clone().requires_grad_(True) for t in (m2, con, torch.cat([col, z[:, None]], -1), op)]
        rc, ra, last = rasterize(*L, None, res, res, 16, offs, flat)
        ((rc * vrc.to(dt)).sum() + (ra * vra.to(dt)).sum()).backward()
        out[dt] = {"L": L, "ra": ra.detach(), "last": last, "offs": offs, "flat": flat, "info": info}
    o64, o32 = out[torch.float64], out[torch.float32]
    ref = {k: t.grad.numpy() for k, t in zip(NAMES, o64["L"], strict=True)}
    ag32 = {k: t.grad.numpy().astype(np.float64) for k, t in zip(NAMES, o32["L"], strict=True)}
    k32 = rasterize_bwd_hipcpu(*[t.detach() for t in o32["L"]], None, res, res, 16, o32["offs"], o32["flat"],
                               o32["ra"], o32["last"], vrc, vra)
    tmp = Path(tempfile.mkdtemp(prefix="bk64_"))
    try:
        f = ctypes.CDLL(str(build_fp64(tmp))).bk_rasterize_bwd_deterministic
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_int] * 7 + [ctypes.c_void_p] * 19
        A = lambda t, d: np.ascontiguousarray(t.detach().numpy(), d)
        fl = A(o64["flat"], np.int32)
        order, seg = segments(fl, n)
        th, tw = o64["offs"].shape
        k64 = {"v_means2d": np.zeros((n, 2)), "v_conics": np.zeros((n, 3)), "v_colors": np.zeros((n, 4)),
               "v_opacities": np.zeros(n)}
        ins = [*(A(t, np.float64) for t in o64["L"]), None, A(o64["offs"], np.int32), fl, A(o64["ra"], np.float64),
               A(o64["last"], np.int32), A(vrc, np.float64), A(vra, np.float64), order, seg]
        P = lambda a: None if a is None else a.ctypes.data_as(ctypes.c_void_p)
        slots = np.zeros(max(1, len(fl) * 12))
        rc = f(n, len(fl), res, res, tw, th, 4, *[P(a) for a in ins], P(slots), *[P(k64[k]) for k in NAMES], None)
        assert rc == 0, rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rows = {}
    for k in NAMES:
        scale = float(np.abs(ref[k]).max())
        rows[k] = {"ref_max_abs": scale}
        for label, got in (("kernel_fp32_division", k32[k]), ("autograd_fp32_cumprod", ag32[k]),
                           ("kernel_fp64_division", k64[k])):
            e = np.abs(got.astype(np.float64) - ref[k])
            rows[k][label] = {"max_abs_err": float(e.max()), "rel_to_ref_max": float(e.max() / scale)}
    return {"n": n, "res": res, "seed": seed, "workload": o64["info"],
            "min_T_final": float((1 - o64["ra"]).min()), "errors_vs_fp64_autograd": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results/renderer/E-R0_precision.json")
    a = ap.parse_args(argv)
    cfg = vars(a)
    res = {"diagnosis": "division-based transmittance recovery (gsplat algorithm) precision", **run_metadata(
           cfg, "benchmarks/renderer/precision_diagnosis.py"), "result": run(a.n, a.res, a.seed)}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["result"]["errors_vs_fp64_autograd"], indent=1))
    print(out)


if __name__ == "__main__":
    main()
