"""Berkelium HIP deterministic backward, executed from its unmodified .hip source under AMD's HIP-CPU runtime.

Validates kernel logic (math, tile ranges, barriers, fixed-order reductions) against (a) PyTorch autograd of the
reference rasterizer in fp32 and fp64 and (b) bitwise repeatability. HIP-CPU execution is NOT GPU evidence;
GPU validation is tests marked ``hip`` (benchmarks/renderer) on a ROCm device.
"""

import numpy as np
import pytest
import torch

from berkelium_neural import camera as cam_mod
from berkelium_neural.contracts import RenderConfig
from berkelium_neural.render.berkelium_ext import BerkeliumRenderer, rasterize_bwd_hipcpu
from berkelium_neural.render.interface import GaussianTensors
from berkelium_neural.render.reference import ReferenceRenderer, isect_tiles, project, rasterize
from berkelium_neural.scene import GaussianScene


def scene(n, seed, spread=0.8):
    rng = np.random.default_rng(seed)
    means = np.c_[rng.uniform(-spread, spread, (n, 2)), 3.0 + rng.uniform(-0.8, 0.8, n)]
    s = GaussianScene(means, np.log(rng.uniform(0.03, 0.25, (n, 3))), rng.normal(size=(n, 4)),
                      rng.uniform(-2, 3, n), rng.normal(size=(n, 3)))
    return s


def two_d_inputs(s, cam, dtype):
    g = GaussianTensors.from_scene(s, dtype=dtype)
    with torch.no_grad():
        m2, con, z, radii, op, col = project(g, cam, RenderConfig())
    offs, flat, info = isect_tiles(m2, radii, z, 16, cam.width, cam.height, 10**7)
    feats = torch.cat([col, z[:, None]], -1)
    leaves = [t.detach().clone().requires_grad_(True) for t in (m2, con, feats, op)]
    return leaves, offs, flat, info


@pytest.mark.parametrize("w,h,n,seed", [(48, 40, 60, 0), (33, 17, 25, 1), (64, 64, 200, 2)])
def test_kernel_matches_autograd(hipcpu_lib, w, h, n, seed):
    cam = cam_mod.make_camera((0.2, -0.1, -0.5), (0, 0, 3), w, h, fov_y_deg=50)
    s = scene(n, seed)
    bg = torch.tensor([0.1, 0.3, 0.2, 0.0])
    ref = {}
    for dt in (torch.float32, torch.float64):
        (m2, con, feats, op), offs, flat, info = two_d_inputs(s, cam, dt)
        rc, ra, last = rasterize(m2, con, feats, op, bg.to(dt), w, h, 16, offs, flat)
        gen = torch.Generator().manual_seed(seed)
        v_rc = torch.randn(h, w, 4, generator=gen).to(dt)
        v_ra = torch.randn(h, w, generator=gen).to(dt)
        (rc * v_rc).sum().add((ra * v_ra).sum()).backward()
        ref[dt] = {"v_means2d": m2.grad, "v_conics": con.grad, "v_colors": feats.grad, "v_opacities": op.grad}
        if dt == torch.float32:
            k = rasterize_bwd_hipcpu(m2, con, feats, op, bg, w, h, 16, offs, flat, ra, last, v_rc, v_ra)
    assert info["n_pairs"] > 0
    for name in ("v_means2d", "v_conics", "v_colors", "v_opacities"):
        got, r32, r64 = torch.from_numpy(k[name]).double(), ref[torch.float32][name].double(), ref[torch.float64][name]
        scale = r64.abs().max().item() + 1e-12
        # kernel (fp32, gsplat division-based T recovery) vs fp64 autograd oracle and vs fp32 autograd
        assert (got - r64).abs().max().item() <= 2e-5 + 2e-3 * scale, name
        assert (got - r32).abs().max().item() <= 2e-5 + 2e-3 * scale, name


def test_kernel_is_bitwise_deterministic(hipcpu_lib):
    w, h = 64, 48
    cam = cam_mod.make_camera((0, 0, -0.5), (0, 0, 3), w, h, fov_y_deg=55)
    (m2, con, feats, op), offs, flat, info = two_d_inputs(scene(400, 7, spread=1.0), cam, torch.float32)
    with torch.no_grad():
        _rc, ra, last = rasterize(m2, con, feats, op, None, w, h, 16, offs, flat)
    v_rc = torch.randn(h, w, 4, generator=torch.Generator().manual_seed(3))
    outs = [rasterize_bwd_hipcpu(m2, con, feats, op, None, w, h, 16, offs, flat, ra, last, v_rc, None)
            for _ in range(5)]
    assert info["max_pairs_per_tile"] > 50          # many Gaussians per tile -> multi-batch + reductions exercised
    for o in outs[1:]:
        for kname in ("v_means2d", "v_conics", "v_colors", "v_opacities"):
            assert o[kname].tobytes() == outs[0][kname].tobytes()


def test_full_renderer_gradients_match_reference(hipcpu_lib):
    """World-space parameter gradients: Berkelium renderer (HIP-CPU backward) vs reference autograd."""
    cam = cam_mod.make_camera((0.1, 0.1, -0.4), (0, 0, 3), 40, 32)
    s = scene(30, 11)
    cfg = RenderConfig(background=(0.05, 0.1, 0.2))
    w = torch.randn(32, 40, 3, generator=torch.Generator().manual_seed(1))
    grads = []
    for r in (ReferenceRenderer(), BerkeliumRenderer("hipcpu")):
        g = GaussianTensors.from_scene(s, requires_grad=True)
        o = r.render(g, cam, cfg)
        ((o.rgb * w).sum() + o.alpha.sum() + 0.05 * o.depth.sum()).backward()
        grads.append([p.grad.clone() for p in g.params()])
        if r.name.startswith("berkelium"):
            assert o.meta["deterministic"] and "HIP-CPU" in r.capabilities()["execution"]
    for a, b in zip(*grads, strict=True):
        assert torch.allclose(a, b, atol=5e-4, rtol=5e-3), (a - b).abs().max()


def test_rocm_path_refuses_without_hip_device():
    from berkelium_neural.render.interface import RendererError
    with pytest.raises(RendererError) as e:
        BerkeliumRenderer("rocm").prepare(RenderConfig())
    assert e.value.code == "DEVICE_UNAVAILABLE"
