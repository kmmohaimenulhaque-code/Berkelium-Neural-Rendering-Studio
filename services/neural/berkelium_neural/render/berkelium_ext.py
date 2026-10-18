"""Berkelium HIP extension: deterministic rasterization backward (csrc/bk_det_bwd.hip).

Two execution paths for the SAME kernel source:
  rocm    torch.utils.cpp_extension build with hipcc on a ROCm PyTorch (real GPU execution, MI300X target).
          Forward = pinned AMD gsplat; backward = Berkelium kernel. Refused on CUDA builds: a CUDA run does not
          validate HIP.
  hipcpu  the unmodified .hip source compiled against AMD's HIP-CPU runtime and called through a C ABI. This
          validates the kernel's logic, barriers and reductions on CPU. It is NOT evidence of GPU execution.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from ..contracts import CameraSpec, RenderConfig, RenderProfile
from .interface import GaussianTensors, RendererError, RenderOutput
from .reference import isect_tiles, project, rasterize

CSRC = Path(__file__).resolve().parents[2] / "csrc"
SOURCES = ("bk_det_bwd.h", "bk_det_bwd.hip")
HIP_CPU_PIN = {"repo": "https://github.com/ROCm/HIP-CPU", "commit": "e112c935057434897bb12d9ab3910380a8bd5f58",
               "license": "MIT"}
HIPCPU_FLAGS = ["-std=c++17", "-O2", "-fPIC", "-shared", "-ffp-contract=off", "-Wno-unknown-pragmas"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def hip_cpu_include() -> Path:
    for c in (os.environ.get("BERKELIUM_HIP_CPU_DIR"), _repo_root() / "third_party" / "hip-cpu"):
        if c and (Path(c) / "include" / "hip" / "hip_runtime.h").exists():
            return Path(c) / "include"
    raise RendererError("DEVICE_UNAVAILABLE", "HIP-CPU runtime not found; run scripts/setup_cpu.sh "
                        f"(clones {HIP_CPU_PIN['repo']}@{HIP_CPU_PIN['commit'][:12]} into third_party/hip-cpu)")


def source_digest() -> str:
    h = hashlib.sha256()
    for f in (*SOURCES, "hipcpu_capi.cpp", "torch_bindings.cpp"):
        h.update((CSRC / f).read_bytes())
    h.update(" ".join(HIPCPU_FLAGS).encode())
    return h.hexdigest()


def build_hipcpu_library(cache_dir: Path | None = None) -> Path:
    inc = hip_cpu_include()
    cache = Path(cache_dir or os.environ.get("BERKELIUM_CACHE", Path.home() / ".cache" / "berkelium"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"libbk_hipcpu_{source_digest()[:16]}.so"
    if lib.exists():
        return lib
    with open(cache / ".build.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if lib.exists():
            return lib
        tmp = lib.with_suffix(".tmp.so")
        cmd = [os.environ.get("CXX", "g++"), *HIPCPU_FLAGS, f"-I{CSRC}", f"-I{inc}", "-x", "c++",
               str(CSRC / "bk_det_bwd.hip"), "-x", "c++", str(CSRC / "hipcpu_capi.cpp"), "-o", str(tmp), "-ltbb",
               "-lpthread"]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RendererError("DEVICE_ERROR", "HIP-CPU build failed:\n" + r.stderr[-4000:])
        tmp.rename(lib)
    return lib


_LIB = None


def _hipcpu():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build_hipcpu_library()))
        f = lib.bk_rasterize_bwd_deterministic
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_int] * 7 + [ctypes.c_void_p] * 19
        _LIB = f
    return _LIB


def _np(t, dtype):
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=dtype) if t is not None else None


def _ptr(a):
    return None if a is None else a.ctypes.data_as(ctypes.c_void_p)


def segments(flatten_ids: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable argsort of flatten_ids + per-Gaussian segment offsets (integer, order-independent)."""
    order = np.argsort(flatten_ids, kind="stable").astype(np.int32)
    counts = np.bincount(flatten_ids.astype(np.int64), minlength=n)
    return order, np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def rasterize_bwd_hipcpu(means2d, conics, colors, opacities, background, width, height, tile_size, isect_offsets,
                         flatten_ids, render_alphas, last_ids, v_render_colors, v_render_alphas, absgrad=False):
    """Run the Berkelium backward kernel under HIP-CPU. Inputs: torch tensors (gsplat layout, one image)."""
    if tile_size != 16:
        raise RendererError("UNSUPPORTED_FEATURE", "Berkelium kernel supports tile_size == 16 only")
    f32, i32 = np.float32, np.int32
    m2, cn, cl, op = _np(means2d, f32), _np(conics, f32), _np(colors, f32), _np(opacities, f32)
    N, C = m2.shape[0], cl.shape[-1]
    fl = _np(flatten_ids, i32)
    order, seg = segments(fl, N)
    th, tw = isect_offsets.shape
    G = 2 + 3 + C + 1 + 2
    out = {"pair_grads": np.zeros(max(1, fl.size * G), f32), "v_means2d": np.zeros((N, 2), f32),
           "v_conics": np.zeros((N, 3), f32), "v_colors": np.zeros((N, C), f32), "v_opacities": np.zeros(N, f32),
           "v_means2d_abs": np.zeros((N, 2), f32) if absgrad else None}
    ins = [m2, cn, cl, op, _np(background, f32), _np(isect_offsets, i32), fl, _np(render_alphas, f32),
           _np(last_ids, i32), _np(v_render_colors, f32), _np(v_render_alphas, f32), order, seg]
    err = _hipcpu()(N, int(fl.size), int(width), int(height), int(tw), int(th), int(C), *[_ptr(a) for a in ins],
                    *[_ptr(out[k]) for k in ("pair_grads", "v_means2d", "v_conics", "v_colors", "v_opacities",
                                              "v_means2d_abs")])
    if err != 0:
        raise RendererError("DEVICE_ERROR", f"HIP-CPU kernel returned hipError {err}")
    return out


# ------------------------------------------------------------------------------------------------- ROCm
_ROCM_EXT = None


def is_rocm_torch() -> bool:
    return bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()


def load_rocm_extension(verbose: bool = False):
    """JIT-build the kernel with hipcc. Requires a ROCm PyTorch with a visible HIP device."""
    global _ROCM_EXT
    if _ROCM_EXT is not None:
        return _ROCM_EXT
    if not getattr(torch.version, "hip", None):
        raise RendererError("DEVICE_UNAVAILABLE", "ROCm PyTorch required (torch.version.hip is None); a CUDA build "
                            "does not validate the HIP kernel")
    if not torch.cuda.is_available():
        raise RendererError("DEVICE_UNAVAILABLE", "no HIP device visible to PyTorch")
    import shutil

    from torch.utils.cpp_extension import load
    # hipify writes translated files next to the sources: build from a private copy, never inside the repo.
    # Flags match gsplat's own build (no -ffp-contract override) so forward/backward threshold decisions agree.
    cache = Path(os.environ.get("BERKELIUM_CACHE", Path.home() / ".cache" / "berkelium"))
    src = cache / f"rocm_src_{source_digest()[:16]}"
    build = src / "build"
    build.mkdir(parents=True, exist_ok=True)
    for f in (*SOURCES, "torch_bindings.cpp"):
        shutil.copy(CSRC / f, src / f)
    _ROCM_EXT = load(name="berkelium_det_bwd", sources=[str(src / "torch_bindings.cpp"), str(src / "bk_det_bwd.hip")],
                     extra_include_paths=[str(src)], extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
                     build_directory=str(build), verbose=verbose)
    return _ROCM_EXT


# ---------------------------------------------------------------------------------------- autograd
class DeterministicRasterize(torch.autograd.Function):
    """2D rasterization with the Berkelium deterministic backward. ``backend`` = 'rocm' | 'hipcpu'."""

    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, background, width, height, tile_size, isect_offsets,
                flatten_ids, backend):
        if backend == "rocm":
            from gsplat.cuda._wrapper import _make_lazy_cuda_func
            rc, ra, last = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")(
                means2d[None], conics[None], colors[None], opacities[None],
                None if background is None else background[None], None, width, height, tile_size,
                isect_offsets[None], flatten_ids)
            rc, ra, last = rc[0], ra[0, ..., 0].float(), last[0]
        else:   # identical gsplat_v1 semantics, computed by the Berkelium reference
            with torch.no_grad():
                rc, ra, last = rasterize(means2d, conics, colors, opacities, background, width, height, tile_size,
                                         isect_offsets, flatten_ids, RenderProfile.GSPLAT_V1)
        ctx.save_for_backward(means2d, conics, colors, opacities, background, isect_offsets, flatten_ids, ra, last)
        ctx.cfg = (width, height, tile_size, backend)
        ctx.mark_non_differentiable(last)
        return rc, ra, last

    @staticmethod
    def backward(ctx, v_rc, v_ra, _v_last):
        means2d, conics, colors, opacities, background, offs, flat, ra, last = ctx.saved_tensors
        W, H, ts, backend = ctx.cfg
        v_ra = torch.zeros_like(ra) if v_ra is None else v_ra
        if backend == "rocm":
            vm, vc, vcol, vo, _ = load_rocm_extension().rasterize_bwd(
                means2d.contiguous(), conics.contiguous(), colors.contiguous(), opacities.contiguous(),
                None if background is None else background.contiguous(), W, H, ts, offs.contiguous(),
                flat.contiguous(), ra.contiguous(), last.contiguous(), v_rc.contiguous(), v_ra.contiguous(), False)
        else:
            o = rasterize_bwd_hipcpu(means2d, conics, colors, opacities, background, W, H, ts, offs, flat, ra, last,
                                     v_rc, v_ra)
            dev, dt = means2d.device, means2d.dtype
            vm, vc, vcol, vo = (torch.from_numpy(o[k]).to(dev, dt)
                                for k in ("v_means2d", "v_conics", "v_colors", "v_opacities"))
        v_bg = None
        if background is not None and ctx.needs_input_grad[4]:
            v_bg = (v_rc * (1.0 - ra)[..., None]).sum((0, 1))
        return vm, vc, vcol, vo, v_bg, None, None, None, None, None, None


class BerkeliumRenderer:
    """AMD gsplat forward + Berkelium deterministic HIP backward (profile gsplat_v1 only)."""

    def __init__(self, backend: str = "auto"):
        self.backend = ("rocm" if is_rocm_torch() else "hipcpu") if backend == "auto" else backend
        if self.backend not in ("rocm", "hipcpu"):
            raise ValueError(backend)
        self.name = f"berkelium_hip[{self.backend}]"

    def capabilities(self) -> dict:
        return {"backend": self.name, "profiles": [RenderProfile.GSPLAT_V1.value], "gradients": "berkelium_hip",
                "deterministic": True, "camera_gradients": False, "tile_size": 16,
                "execution": "gpu" if self.backend == "rocm" else "cpu_emulated_hip (HIP-CPU; not GPU evidence)"}

    def prepare(self, cfg: RenderConfig) -> str:
        if cfg.profile != RenderProfile.GSPLAT_V1:
            raise RendererError("UNSUPPORTED_FEATURE", "berkelium_hip implements profile gsplat_v1 only")
        if self.backend == "hipcpu":
            build_hipcpu_library()
            return "cpu"
        load_rocm_extension()
        return "cuda"

    def render(self, g: GaussianTensors, cam: CameraSpec, cfg: RenderConfig) -> RenderOutput:
        if cfg.profile != RenderProfile.GSPLAT_V1:
            raise RendererError("UNSUPPORTED_FEATURE", "berkelium_hip implements profile gsplat_v1 only")
        if cfg.tile_size != 16:
            raise RendererError("UNSUPPORTED_FEATURE", "tile_size must be 16")
        means2d, conics, depths, radii, opac, colors = project(g, cam, cfg)
        offs, flat, info = isect_tiles(means2d, radii, depths, 16, cam.width, cam.height, cfg.max_pairs)
        dev = means2d.device
        feats = torch.cat([colors, depths[:, None]], -1)
        bg = torch.tensor([*cfg.background, 0.0], dtype=means2d.dtype, device=dev)
        rc, ra, last = DeterministicRasterize.apply(means2d, conics, feats, opac, bg, cam.width, cam.height, 16,
                                                    offs.to(dev), flat.to(dev), self.backend)
        depth = rc[..., 3] / ra.clamp(min=1e-10)
        meta = {"backend": self.name, "profile": cfg.profile.value, "deterministic": True,
                "n_gaussians": int(g.means.shape[0]), "n_visible": int((radii > 0).all(-1).sum()), **info}
        return RenderOutput(rgb=rc[..., :3], alpha=ra, depth=depth, meta=meta,
                            aux={"isect_offsets": offs, "flatten_ids": flat, "last_ids": last})
