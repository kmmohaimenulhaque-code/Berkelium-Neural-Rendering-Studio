"""M7 fitting building blocks and the M6 TripoSR CPU marching-cubes shim, on small synthetic inputs (seconds, CPU)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from berkelium_neural.camera import orbit_cameras
from berkelium_neural.fitting import real_object as ro

ROOT = Path(__file__).resolve().parents[3]


def _sphere_scan(n=24):
    """UV sphere (radius 0.6) textured with a left-red / right-blue texture: a ScanMesh without any download."""
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n), indexing="ij")
    v = 0.6 * np.stack([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)], -1).reshape(-1, 3)
    uv = np.stack([ph / (2 * np.pi), 1 - th / np.pi], -1).reshape(-1, 2)
    f = []
    W = 2 * n
    for i in range(n - 1):
        for j in range(W - 1):
            a, b, c, d = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            f += [[a, c, b], [b, c, d]]
    tex = np.zeros((64, 64, 3), np.float32)
    tex[:, :32] = (1.0, 0.1, 0.1)
    tex[:, 32:] = (0.1, 0.2, 1.0)
    return ro.ScanMesh(v=v, f=np.array(f), uv=uv, tex=tex, transform={}, source={})


def test_render_scan_is_textured_and_has_soft_silhouette():
    scan = _sphere_scan()
    cam = orbit_cameras(1, 3.0, 0.0, 48, 48, 40.0)[0]
    rgb, a = ro.render_scan(scan, cam)
    assert rgb.shape == (48, 48, 3) and a.shape == (48, 48)
    assert a[24, 24] == 1.0 and a[0, 0] == 0.0 and ((a > 0) & (a < 1)).any()     # supersampled edges
    inside = rgb[a > 0.99]
    assert (inside.max(1) > 0.5).all()                                           # texture colours, not black
    assert np.allclose(rgb[a == 0], 0)                                           # premultiplied on black


def test_space_carving_recovers_the_sphere_from_silhouettes_only():
    scan = _sphere_scan()
    cams, _ = ro.view_split(48, n_train=12, n_test=2)
    alphas = [ro.render_scan(scan, c)[1] for c in cams]
    pts, vs, occ = ro.space_carve(cams, alphas, 2000, grid=40)
    r = np.linalg.norm(pts, axis=1)
    assert occ > 0 and abs(np.median(r) - 0.6) < 0.08, np.median(r)            # carved hull hugs the surface


def test_tiny_fit_improves_heldout_views_and_geometry_metrics_run():
    scan = _sphere_scan()
    train, test = ro.view_split(32, n_train=8, n_test=2)
    tr = [ro.render_scan(scan, c) for c in train]
    te = [ro.render_scan(scan, c) for c in test]
    init, fitted, hist = ro.fit(train, [x[0] for x in tr], [x[1] for x in tr], n=600, steps=60, log=lambda *a: None)
    assert hist[-1]["loss"] < hist[0]["loss"]
    m0 = ro.image_metrics(*ro.render_scene(init, test), [x[0] for x in te], [x[1] for x in te])
    m1 = ro.image_metrics(*ro.render_scene(fitted, test), [x[0] for x in te], [x[1] for x in te])
    assert m1["psnr"] > m0["psnr"], (m0["psnr"], m1["psnr"])
    g = ro.geometry_metrics(fitted, scan, n_surface=2000)
    assert g["n_opaque"] > 0 and g["chamfer_l1"] < 0.2


def test_gso_pins_refuse_unknown_objects(tmp_path):
    with pytest.raises(ValueError):
        ro.fetch_gso("Not_A_Pinned_Object", tmp_path)


def test_triposr_marching_cubes_shim_matches_torchmcubes_convention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("skimage")
    sys.path.insert(0, str(ROOT / "scripts"))
    import m6_triposr_controls as m6
    m6.install_shims()
    from torchmcubes import marching_cubes
    R = 32
    ax = torch.linspace(0, 1, R)
    x, y, z = torch.meshgrid(ax, ax, ax, indexing="ij")
    # an off-centre blob along the FIRST grid axis: after TripoSR's [2, 1, 0] reorder it must sit along x
    level = 0.2 - torch.sqrt((x - 0.7) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2)
    v, f = marching_cubes(level, 0.0)
    v = v[:, [2, 1, 0]] / (R - 1)                      # exactly what tsr/models/isosurface.py does
    c = v.mean(0)
    assert abs(float(c[0]) - 0.7) < 0.03 and abs(float(c[1]) - 0.5) < 0.03 and abs(float(c[2]) - 0.5) < 0.03
    assert f.dtype == torch.int64 and len(f) > 100
    with pytest.raises(RuntimeError):
        sys.modules["rembg"].remove(None)


def test_depth_paths_agree_on_a_known_sphere():
    """Scan depth and Gaussian rendered depth both measure camera Z; a radius-0.6 sphere seen from distance 3 has
    its front surface at Z = 2.4 in the image centre."""
    scan = _sphere_scan(32)
    cam = orbit_cameras(1, 3.0, 0.0, 48, 48, 40.0)[0]
    _, a, d = ro.render_scan(scan, cam, with_depth=True)
    assert abs(float(d[24, 24]) - 2.4) < 0.01 and d[0, 0] == 0
    m = ro.depth_metrics([d], [a], [d], [a])
    assert m["mean_abs"] == 0 and m["within_0.02"] == 1.0 and m["coverage"] == 1.0
    m2 = ro.depth_metrics([d + 0.03], [a], [d], [a])
    assert abs(m2["mean_abs"] - 0.03) < 1e-5 and m2["within_0.02"] == 0.0 and m2["within_0.05"] == 1.0
