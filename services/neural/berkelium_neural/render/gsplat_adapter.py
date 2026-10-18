"""AMD gsplat (ROCm) backend adapter — the differentiable-rendering foundation.

Pinned upstream: AMD-Ecosystem/gsplat tag v1.5.3b2 (commit ff8bc5489545bcc1b5a2b997d975834429a0d7af);
Apache-2.0 upstream portions + MIT AMD modifications (see NOTICE.txt in that repository). Supported: ROCm 6.4.3 /
7.0.0, PyTorch 2.6 / 2.8, Python 3.10 / 3.12, MI300X.

``raster_backward`` selects the gradient path for the SAME gsplat forward:
  "gsplat"     upstream rasterize_to_pixels_3dgs_bwd (float atomicAdd accumulation across tiles)
  "berkelium"  Berkelium deterministic HIP kernel (csrc/bk_det_bwd.hip)
Everything else (projection, tile intersection, sort, forward compositing) is identical, so differences between
the two are attributable to the backward accumulation alone.
"""

from __future__ import annotations

import math

import torch

from ..camera import K as Kmat
from ..camera import T_cw
from ..contracts import CameraSpec, RenderConfig, RenderProfile
from ..scene import LOG_SCALE_MAX, LOG_SCALE_MIN
from .interface import GaussianTensors, RendererError, RenderOutput

GSPLAT_PIN = {"repo": "https://github.com/AMD-Ecosystem/gsplat", "tag": "v1.5.3b2",
              "commit": "ff8bc5489545bcc1b5a2b997d975834429a0d7af", "license": "Apache-2.0 (upstream) + MIT (AMD)"}


def gsplat_status() -> dict:
    """What is actually available: package import, version, ROCm build, visible device. Never inferred."""
    st = {"pin": GSPLAT_PIN, "torch": torch.__version__, "torch_hip": getattr(torch.version, "hip", None),
          "torch_cuda": getattr(torch.version, "cuda", None), "device_visible": torch.cuda.is_available()}
    try:
        import gsplat
        st["gsplat_version"] = getattr(gsplat, "__version__", None)
        st["gsplat_file"] = gsplat.__file__
    except Exception as e:  # noqa: BLE001
        st["gsplat_import_error"] = f"{type(e).__name__}: {e}"
    st["usable"] = bool(st.get("gsplat_version") and st["torch_hip"] and st["device_visible"])
    return st


class GsplatRenderer:
    def __init__(self, raster_backward: str = "gsplat", require_rocm: bool = True):
        if raster_backward not in ("gsplat", "berkelium"):
            raise ValueError(raster_backward)
        self.raster_backward = raster_backward
        self.require_rocm = require_rocm
        self.name = "gsplat" if raster_backward == "gsplat" else "gsplat+berkelium_bwd"

    def capabilities(self) -> dict:
        st = gsplat_status()
        return {"backend": self.name, "profiles": [RenderProfile.GSPLAT_V1.value], "pin": GSPLAT_PIN,
                "deterministic": self.raster_backward == "berkelium", "usable": st["usable"], "status": st}

    def prepare(self, cfg: RenderConfig) -> str:
        """Validate availability BEFORE any allocation; returns the torch device to place tensors on."""
        self._check(cfg)
        return "cuda"

    def _check(self, cfg: RenderConfig):
        st = gsplat_status()
        if "gsplat_import_error" in st:
            raise RendererError("DEVICE_UNAVAILABLE", f"gsplat not importable: {st['gsplat_import_error']}")
        if self.require_rocm and not st["torch_hip"]:
            raise RendererError("DEVICE_UNAVAILABLE", "ROCm PyTorch required; a CUDA build does not validate the AMD "
                                "HIP port (pass require_rocm=False only for an explicitly labelled CUDA run)")
        if not st["device_visible"]:
            raise RendererError("DEVICE_UNAVAILABLE", "no GPU device visible to PyTorch")
        if cfg.profile != RenderProfile.GSPLAT_V1:
            raise RendererError("UNSUPPORTED_FEATURE", "gsplat implements profile gsplat_v1 only")

    def render(self, g: GaussianTensors, cam: CameraSpec, cfg: RenderConfig) -> RenderOutput:
        self._check(cfg)
        from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles, rasterize_to_pixels
        dev = g.means.device
        if dev.type != "cuda":
            raise RendererError("INVALID_ARGUMENT", "Gaussian tensors must be on the GPU device")
        means, quats = g.means.float(), g.quats.float()
        if (quats.norm(dim=-1) < 1e-8).any():
            raise RendererError("INVALID_QUATERNION", "quaternion norm < 1e-8")
        scales = torch.exp(g.log_scales.float().clamp(LOG_SCALE_MIN, LOG_SCALE_MAX))
        opac = torch.sigmoid(g.opacity_logits.float())
        colors = torch.sigmoid(g.color_logits.float())
        V = torch.tensor(T_cw(cam), dtype=torch.float32, device=dev)[None]
        Kt = torch.tensor(Kmat(cam), dtype=torch.float32, device=dev)[None]
        W, H, ts = cam.width, cam.height, cfg.tile_size
        radii, means2d, depths, conics, _ = fully_fused_projection(
            means, None, quats, scales, V, Kt, W, H, eps2d=cfg.eps2d, near_plane=cam.near, far_plane=cam.far,
            opacities=opac)
        tw, th = math.ceil(W / ts), math.ceil(H / ts)
        _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, ts, tw, th)
        n_pairs = int(flatten_ids.numel())
        if n_pairs > cfg.max_pairs:
            raise RendererError("RESOURCE_LIMIT", f"{n_pairs} pairs > max_pairs={cfg.max_pairs}")
        offsets = isect_offset_encode(isect_ids, 1, tw, th)
        feats = torch.cat([colors[None].expand(1, -1, 3), depths[..., None]], -1)          # RGB + ED channel
        bg = torch.tensor([[*cfg.background, 0.0]], dtype=torch.float32, device=dev)
        if self.raster_backward == "gsplat":
            rc, ra = rasterize_to_pixels(means2d, conics, feats, opac[None].expand(1, -1), W, H, ts, offsets,
                                         flatten_ids, backgrounds=bg)
            rc, ra = rc[0], ra[0, ..., 0]
        else:
            from .berkelium_ext import DeterministicRasterize, is_rocm_torch
            if not is_rocm_torch():
                raise RendererError("DEVICE_UNAVAILABLE", "Berkelium HIP backward requires ROCm")
            rc, ra, _ = DeterministicRasterize.apply(means2d[0], conics[0], feats[0], opac, bg[0], W, H, ts,
                                                     offsets[0], flatten_ids, "rocm")
        depth = rc[..., 3] / ra.clamp(min=1e-10)
        meta = {"backend": self.name, "profile": cfg.profile.value, "deterministic": self.raster_backward ==
                "berkelium", "device": torch.cuda.get_device_name(dev), "n_gaussians": int(means.shape[0]),
                "n_visible": int((radii[0] > 0).all(-1).sum()), "n_pairs": n_pairs,
                "max_pairs_per_tile": int(torch.diff(torch.cat([offsets.flatten(), offsets.new_tensor([n_pairs])]))
                                          .max()) if n_pairs else 0, "pin": GSPLAT_PIN["commit"]}
        return RenderOutput(rgb=rc[..., :3], alpha=ra, depth=depth, meta=meta,
                            aux={"means2d": means2d[0], "conics": conics[0], "radii": radii[0], "depths": depths[0],
                                 "isect_offsets": offsets[0], "flatten_ids": flatten_ids})


def get_renderer(name: str):
    """Typed factory. Raises RendererError when the requested backend is not actually available."""
    if name == "reference_cpu":
        from .reference import ReferenceRenderer
        return ReferenceRenderer()
    if name == "gsplat":
        return GsplatRenderer("gsplat")
    if name == "berkelium_hip":
        from .berkelium_ext import BerkeliumRenderer, is_rocm_torch
        return GsplatRenderer("berkelium") if is_rocm_torch() else BerkeliumRenderer("hipcpu")
    raise RendererError("INVALID_ARGUMENT", f"unknown renderer {name!r}")
