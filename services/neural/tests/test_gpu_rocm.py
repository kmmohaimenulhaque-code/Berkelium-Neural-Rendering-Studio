"""ROCm-only validation (MI300X). Skipped unless a ROCm PyTorch with a visible HIP device AND gsplat are present.
A CUDA device does NOT satisfy these tests: the HIP port and the Berkelium HIP kernel are what is under test."""

import hashlib

import pytest
import torch

from berkelium_neural import camera as cam_mod
from berkelium_neural.contracts import RenderConfig
from berkelium_neural.render.interface import GaussianTensors

pytestmark = [pytest.mark.hip, pytest.mark.gpu]


def _rocm():
    if not (getattr(torch.version, "hip", None) and torch.cuda.is_available()):
        pytest.skip("requires ROCm PyTorch with a visible HIP device")
    try:
        import gsplat  # noqa: F401
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"gsplat not importable: {e}")


def _scene(n=3000, seed=0):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from benchmarks.renderer.bench_renderer import synthetic_scene
    return synthetic_scene(n, seed)


def _fb(r, s, cam, dev, dt):
    g = GaussianTensors.from_scene(s, device=dev, dtype=dt, requires_grad=True)
    o = r.render(g, cam, RenderConfig())
    w = torch.randn(cam.height, cam.width, 3, generator=torch.Generator().manual_seed(0)).to(dev, dt)
    ((o.rgb * w).sum() + o.alpha.sum()).backward()
    return o, [p.grad.detach().double().cpu() for p in g.params()]


def test_gsplat_forward_matches_reference():
    _rocm()
    from berkelium_neural.render.gsplat_adapter import GsplatRenderer
    from berkelium_neural.render.reference import ReferenceRenderer
    s, cam = _scene(500), cam_mod.make_camera((0.2, -0.1, -0.6), (0, 0, 3), 96, 96, fov_y_deg=55)
    og, _ = _fb(GsplatRenderer("gsplat"), s, cam, "cuda", torch.float32)
    orf, _ = _fb(ReferenceRenderer(), s, cam, "cpu", torch.float64)
    assert (og.rgb.double().cpu() - orf.rgb).abs().max() < 2e-4
    assert (og.alpha.double().cpu() - orf.alpha).abs().max() < 2e-4


def test_berkelium_kernel_matches_gsplat_and_reference_gradients():
    _rocm()
    from berkelium_neural.render.gsplat_adapter import GsplatRenderer
    from berkelium_neural.render.reference import ReferenceRenderer
    s, cam = _scene(500), cam_mod.make_camera((0.2, -0.1, -0.6), (0, 0, 3), 96, 96, fov_y_deg=55)
    _, gg = _fb(GsplatRenderer("gsplat"), s, cam, "cuda", torch.float32)
    _, gb = _fb(GsplatRenderer("berkelium"), s, cam, "cuda", torch.float32)
    _, gr = _fb(ReferenceRenderer(), s, cam, "cpu", torch.float64)
    for a, b, r in zip(gg, gb, gr, strict=True):
        tol = 2e-5 + 2e-3 * r.abs()
        assert ((b - r).abs() <= tol).all()
        assert ((b - a).abs() <= tol).all()


def test_berkelium_kernel_bitwise_deterministic_on_gpu():
    _rocm()
    from berkelium_neural.render.gsplat_adapter import GsplatRenderer
    s, cam = _scene(20000, 1), cam_mod.make_camera((0, 0, -0.6), (0, 0, 3), 256, 256, fov_y_deg=55)
    hashes = {hashlib.sha256(b"".join(x.numpy().tobytes() for x in _fb(GsplatRenderer("berkelium"), s, cam, "cuda",
                                                                         torch.float32)[1])).hexdigest()
              for _ in range(10)}
    assert len(hashes) == 1
