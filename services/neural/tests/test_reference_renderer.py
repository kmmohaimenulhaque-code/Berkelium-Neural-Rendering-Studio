"""CPU renderer semantics: analytic fixtures, camera math, scene format, gsplat parity, FD gradient checks."""

import math

import numpy as np
import pytest
import torch

from berkelium_neural import camera as cam_mod
from berkelium_neural.contracts import CameraSpec, RenderConfig, RenderProfile
from berkelium_neural.render.interface import GaussianTensors, RendererError
from berkelium_neural.render.reference import ReferenceRenderer, project
from berkelium_neural.scene import GaussianScene, SceneError


def logit(p):
    return math.log(p / (1 - p))


def front_cam(w=65, h=65, f=64.0):
    I = tuple(tuple(float(v) for v in r) for r in np.eye(4))
    return CameraSpec(width=w, height=h, fx=f, fy=f, cx=w / 2, cy=h / 2, T_cw=I)


def gauss(means, scale=0.1, opac=0.5, rgb=(0.25, 0.5, 0.75), dtype=torch.float64):
    n = len(means)
    s = GaussianScene(np.array(means), np.full((n, 3), math.log(scale)), np.tile([1, 0, 0, 0], (n, 1)),
                      np.full(n, logit(opac)), np.tile([logit(c) for c in rgb], (n, 1)))
    return GaussianTensors.from_scene(s, dtype=dtype)


R = ReferenceRenderer()


@pytest.mark.parametrize("profile", list(RenderProfile))
def test_single_gaussian_center_pixel(profile):
    out = R.render(gauss([[0, 0, 2.0]]), front_cam(), RenderConfig(profile=profile))
    c = out.rgb[32, 32]
    assert torch.allclose(out.alpha[32, 32], torch.tensor(0.5, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(c, torch.tensor([0.125, 0.25, 0.375], dtype=torch.float64), atol=1e-12)
    assert abs(float(out.depth[32, 32]) - 2.0) < 1e-12
    # symmetric footprint about the principal point
    assert torch.allclose(out.alpha, out.alpha.flip(0), atol=1e-12) and torch.allclose(out.alpha, out.alpha.flip(1))


@pytest.mark.parametrize("profile", list(RenderProfile))
def test_two_coincident_gaussians_compositing(profile):
    g = gauss([[0, 0, 2.0], [0, 0, 3.0]], rgb=(0.9, 0.1, 0.1))
    g.color_logits.data[1] = torch.tensor([logit(0.1), logit(0.1), logit(0.9)], dtype=torch.float64)
    out = R.render(g, front_cam(), RenderConfig(profile=profile))
    assert abs(float(out.alpha[32, 32]) - 0.75) < 1e-12
    expect = 0.5 * torch.tensor([0.9, 0.1, 0.1]) + 0.25 * torch.tensor([0.1, 0.1, 0.9])
    assert torch.allclose(out.rgb[32, 32], expect.double(), atol=1e-12)
    assert abs(float(out.depth[32, 32]) - 7.0 / 3.0) < 1e-12


def test_empty_and_culled_scene_is_background():
    cfg = RenderConfig(background=(0.1, 0.2, 0.3))
    out = R.render(gauss([[0, 0, -2.0]]), front_cam(), cfg)            # behind the camera -> culled
    assert out.meta["n_visible"] == 0 and out.meta["n_pairs"] == 0
    assert torch.all(out.alpha == 0)
    assert torch.allclose(out.rgb, torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64).expand_as(out.rgb))


def test_gsplat_exclusive_vs_spec_inclusive_termination():
    """Ten alpha-0.95 layers (T after k layers = 0.05^k, never exactly on the 1e-4 boundary):
    gsplat applies layers while T_next > 1e-4 (3 layers, the 4th would give 6.25e-6 -> excluded);
    the spec stops only once T_before < 1e-4 (4th layer at T_before=1.25e-4 is included)."""
    g = gauss([[0, 0, 2.0 + 0.1 * i] for i in range(10)], opac=0.95, scale=0.3)
    a_g = float(R.render(g, front_cam(), RenderConfig(profile=RenderProfile.GSPLAT_V1)).alpha[32, 32])
    a_s = float(R.render(g, front_cam(), RenderConfig(profile=RenderProfile.BERKELIUM_V02)).alpha[32, 32])
    assert abs(a_g - (1 - 0.05 ** 3)) < 1e-6
    assert abs(a_s - (1 - 0.05 ** 4)) < 1e-6


def test_resource_limit_is_typed():
    with pytest.raises(RendererError) as e:
        R.render(gauss([[0, 0, 2.0]], scale=0.5), front_cam(), RenderConfig(max_pairs=1))
    assert e.value.code == "RESOURCE_LIMIT"


def test_camera_conventions():
    T = cam_mod.look_at((0, 0, 5), (0, 0, 0))
    assert np.allclose(T[:3, :3], np.diag([-1, -1, -1]) @ np.diag([-1, 1, 1]))   # x right, y down, z toward -Z
    c = cam_mod.make_camera((0, 0, 5), width=64, height=48)
    uv, z = cam_mod.project_points(c, np.array([[0, 0, 0.0], [0, 1.0, 0]]))
    assert np.allclose(uv[0], [32, 24]) and z[0] == pytest.approx(5)
    assert uv[1, 1] < uv[0, 1]                                                    # world +Y is image up
    gl = np.eye(4)
    gl[:3, 3] = (0, 0, 5)                                                         # OpenGL cam at +Z looking -Z
    c2 = cam_mod.from_opengl_c2w(gl, 64, 48, c.fx, c.fy, 32, 24)
    assert np.allclose(np.asarray(c2.T_cw), np.asarray(c.T_cw), atol=1e-12)
    r = cam_mod.resized(c, 32, 24)
    assert r.fx == pytest.approx(c.fx / 2) and r.cx == pytest.approx(16)
    with pytest.raises(ValueError):
        CameraSpec(width=4, height=4, fx=1, fy=1, cx=2, cy=2, T_cw=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0),
                                                                     (0, 0, 0, 1)))  # reflection rejected


def test_scene_roundtrip_and_corruption():
    rng = np.random.default_rng(0)
    s = GaussianScene(rng.normal(size=(7, 3)), rng.normal(size=(7, 3)) * 0.1, rng.normal(size=(7, 4)),
                      rng.normal(size=7), rng.normal(size=(7, 3)), meta={"units": "normalized_object"})
    payload, man = s.to_bytes()
    s2 = GaussianScene.from_bytes(payload, man)
    assert np.array_equal(s2.means, s.means) and np.array_equal(s2.ids, s.ids) and s2.meta == s.meta
    bad = bytearray(payload)
    bad[300] ^= 1
    with pytest.raises(SceneError):
        GaussianScene.from_bytes(bytes(bad), man)
    with pytest.raises(SceneError):
        GaussianScene(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 4)), np.zeros(1), np.zeros((1, 3))).validate()
    assert s.to_3dgs_ply().startswith(b"ply\n")


def random_scene(n, seed, dtype=torch.float64, spread=0.6, z0=3.0):
    rng = np.random.default_rng(seed)
    means = np.c_[rng.uniform(-spread, spread, (n, 2)), z0 + rng.uniform(-0.5, 0.5, n)]
    s = GaussianScene(means, np.log(rng.uniform(0.05, 0.2, (n, 3))), rng.normal(size=(n, 4)),
                      rng.uniform(-1, 2, n), rng.normal(size=(n, 3)))
    return GaussianTensors.from_scene(s, dtype=dtype)


def test_projection_matches_pinned_gsplat_torch_reference(gsplat_torch_impl):
    """Berkelium projection vs AMD gsplat 1.5.3b2 `_fully_fused_projection` (pure PyTorch, CPU)."""
    from berkelium_neural.render.interface import covariance
    g = random_scene(64, 1, spread=1.5)
    c = cam_mod.make_camera((0.3, 0.2, -2.0), (0, 0, 3), 80, 60)
    m2, conics, z, radii, *_ = project(g, c, RenderConfig())
    means, q, s, *_ = g.activated()
    V = torch.tensor(cam_mod.T_cw(c))[None]
    Kt = torch.tensor(cam_mod.K(c))[None]
    gr, gm2, gz, gcon, _ = gsplat_torch_impl._fully_fused_projection(means, covariance(q, s), V, Kt, 80, 60, eps2d=0.3,
                                                                     near_plane=c.near, far_plane=c.far)
    vis = (radii > 0).all(-1) & (gr[0] > 0).all(-1)
    assert vis.sum() > 20
    assert torch.allclose(m2[vis], gm2[0][vis], atol=1e-9)
    assert torch.allclose(conics[vis], gcon[0][vis], rtol=1e-9, atol=1e-12)
    assert torch.allclose(z, gz[0], atol=1e-12)


def test_render_is_bitwise_deterministic():
    g = random_scene(40, 2, dtype=torch.float32)
    c = cam_mod.make_camera((0, 0, -1.0), (0, 0, 3), 48, 40)
    a = R.render(g, c, RenderConfig())
    b = R.render(g, c, RenderConfig())
    assert torch.equal(a.rgb, b.rgb) and torch.equal(a.alpha, b.alpha) and torch.equal(a.depth, b.depth)


def test_finite_difference_gradients_all_parameter_families():
    """FP64 central differences vs autograd for every raw parameter family (smooth fixture: well-separated
    depths, no threshold crossings within the step)."""
    g = random_scene(4, 3, dtype=torch.float64, spread=0.25)
    c = cam_mod.make_camera((0, 0, -0.5), (0, 0, 3), 20, 18)
    cfg = RenderConfig(background=(0.2, 0.1, 0.3))
    w = torch.randn(18, 20, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64)

    def loss(*params):
        gg = GaussianTensors(*params)
        o = R.render(gg, c, cfg)
        return (torch.cat([o.rgb, o.alpha[..., None], o.depth[..., None] * 0.1], -1) * w).sum()

    params = [p.detach().clone().requires_grad_(True) for p in g.params()]
    assert torch.autograd.gradcheck(loss, params, eps=1e-6, atol=1e-6, rtol=1e-4)
