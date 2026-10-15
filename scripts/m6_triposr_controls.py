#!/usr/bin/env python3
"""M6 / research stage C: TripoSR single-image reconstruction on Berkelium-owned control images, compared with
exact ground truth and with Berkelium's own text-to-3D model, then consumed by the Gaussian pipeline.

  .venv-triposr/bin/python scripts/m6_triposr_controls.py          # needs data/procedural_shapes_v1 (test split)

Pins (TEXT_TO_3D_RESEARCH_PLAN.md [R6], NEURAL_DATA_PROVENANCE.md): code VAST-AI-Research/TripoSR @ 107cefdc244c...
(third_party/TripoSR), weights stabilityai/TripoSR @ 5b521936b01f... (MIT per upstream). Separate environment
(.venv-triposr): TripoSR's own pins are old; only its model code is used.

Compatibility branch (D-0xx, research plan stage C): torchmcubes (a CUDA extension) is replaced by scikit-image
marching cubes on CPU via a shim with torchmcubes' axis convention; rembg is not used (control images come with
exact alpha), so a stub that raises is installed. Control images are renders of Berkelium-owned procedural assets
(MIT) from the frozen test split, so ground truth geometry is exact and independent of both models.

Gates (stage C): finite mesh, non-empty faces, bounds/units metadata, renderable 24-view orbit, time per input
after load, host peak memory; recognisability is a recorded visual review, not a metric. Comparison: Chamfer-L1 /
F-score vs GT after the same similarity alignment for every candidate. Outputs: results/m6/.
"""
import argparse
import json
import resource
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "neural"))
sys.path.insert(0, str(ROOT / "scripts"))
TRIPOSR = ROOT / "third_party" / "TripoSR"
CODE_PIN = "107cefdc244c39106fa830359024f6a2f1c78871"
WEIGHTS_PIN = "5b521936b01fbe1890f6f9baed0254ab6351c04a"
CONTROLS = ["on_top_of__cylinder_sphere__000", "next_to__box_box__000", "next_to__torus_torus__000"]


def install_shims():
    """torchmcubes -> scikit-image marching cubes (CPU); rembg -> stub that refuses to run."""
    import torch
    from skimage.measure import marching_cubes as sk_mc

    def marching_cubes(vol, thresh):
        v, f, _, _ = sk_mc(vol.detach().cpu().numpy().astype(np.float32), level=float(thresh))
        v = np.ascontiguousarray(v[:, ::-1])          # torchmcubes returns (z, y, x); TripoSR reorders [2, 1, 0]
        return torch.from_numpy(v.astype(np.float32)), torch.from_numpy(f.astype(np.int64))
    sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=marching_cubes)

    def _no_rembg(*a, **k):
        raise RuntimeError("rembg is not used in this pipeline: control images carry exact alpha")
    sys.modules["rembg"] = types.SimpleNamespace(new_session=_no_rembg, remove=_no_rembg)


def control_image(rec, root, size=512, ratio=0.85):
    """Owned control image: shaded render of the GT mesh (3/4 view, elevation 15 deg), foreground cropped to
    ``ratio`` of a square frame and composited on 50 % grey (TripoSR's documented preprocessing)."""
    from PIL import Image

    from berkelium_neural.camera import make_camera
    from berkelium_neural.text3d.geometry import render_mesh
    gt = np.load(root / rec.files["gt_mesh"])
    cam = make_camera((2.2, 0.9, 2.2), (0, 0, 0), 768, 768, fov_y_deg=40)
    rgb, mask, _ = render_mesh(gt["v"], gt["f"], gt["c"], cam, background=(0, 0, 0))
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, a = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1].astype(np.float32)
    side = int(max(crop.shape[:2]) / ratio)
    canvas = np.full((side, side, 3), 0.5, np.float32)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop * a[..., None] + 0.5 * (1 - a[..., None])
    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).round().astype(np.uint8)).resize((size, size),
                                                                                       Image.Resampling.LANCZOS)
    return img, cam


def normalise(v):
    lo, hi = v.min(0), v.max(0)
    return (v - (lo + hi) / 2) / float(np.linalg.norm(hi - lo))


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def align(src_pts, dst_pts, iters=30):
    """Similarity alignment (rotation + uniform scale + translation) of src to dst: 24 axis-aligned starting
    rotations, each refined by ICP with Umeyama; the best final Chamfer wins. Same procedure for every candidate."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst_pts)
    starts = []
    for ax in range(3):
        for flip in (0, 180):
            for yaw in (0, 90, 180, 270):
                starts.append(_rot(1, yaw) @ _rot(ax, flip) if ax != 1 else _rot(1, yaw) @ _rot(0, flip + 90))
    best = None
    for R0 in starts:
        R, s, t = R0, 1.0, np.zeros(3)
        for _ in range(iters):
            cur = s * src_pts @ R.T + t
            _, idx = tree.query(cur)
            m = dst_pts[idx]
            mu_s, mu_d = src_pts.mean(0), m.mean(0)
            A, B = src_pts - mu_s, m - mu_d
            U, S, Vt = np.linalg.svd(B.T @ A / len(A))
            D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
            R = U @ D @ Vt
            s = float(np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean())
            t = mu_d - s * R @ mu_s
        cur = s * src_pts @ R.T + t
        d1, _ = tree.query(cur)
        d2, _ = cKDTree(cur).query(dst_pts)
        cd = 0.5 * (d1.mean() + d2.mean())
        if best is None or cd < best[0]:
            best = (cd, R, s, t)
    return best


def compare(v, f, gt_v, gt_f, aligned: bool):
    from scipy.spatial import cKDTree

    from berkelium_neural.text3d.geometry import sample_surface
    gv = normalise(gt_v)
    p, _, _ = sample_surface(normalise(v), f, 8000, 1)
    q, _, _ = sample_surface(gv, gt_f, 8000, 2)
    note = "native frame, bbox-normalised"
    if aligned:
        _, R, s, t = align(p, q)
        p = s * p @ R.T + t
        note = "similarity-aligned to GT (ICP, 24 starts)"
    d1, _ = cKDTree(q).query(p)
    d2, _ = cKDTree(p).query(q)
    out = {"chamfer_l1": float(0.5 * (d1.mean() + d2.mean())), "frame": note, "units": "fraction of bbox diagonal"}
    for th in (0.02, 0.05):
        pr, rc = float((d1 < th).mean()), float((d2 < th).mean())
        out[f"fscore@{th}"] = 2 * pr * rc / max(pr + rc, 1e-12)
    return out


def orbit_sheet(v, f, col, n=24, res=96):
    from berkelium_neural.camera import orbit_cameras
    from berkelium_neural.text3d.geometry import render_mesh
    vn = normalise(v) * 1.6
    cams = orbit_cameras(n, 3.2, 20.0, res, res)
    tiles = [render_mesh(vn, f, col, c)[0] for c in cams]
    rows = [np.concatenate(tiles[i:i + 6], 1) for i in range(0, n, 6)]
    return np.concatenate(rows, 0), tiles


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(ROOT / "data" / "procedural_shapes_v1"))
    ap.add_argument("--controls", nargs="+", default=CONTROLS)
    ap.add_argument("--mc-res", type=int, default=128)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "m6"))
    a = ap.parse_args(argv)
    import subprocess

    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(TRIPOSR), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if head != CODE_PIN:
        raise SystemExit(f"third_party/TripoSR is at {head}, not the pinned {CODE_PIN}")
    install_shims()
    sys.path.insert(0, str(TRIPOSR))
    from tsr.system import TSR
    from tsr.utils import to_gradio_3d_orientation

    from _common import provenance
    from berkelium_neural.contracts import RenderConfig
    from berkelium_neural.render.interface import GaussianTensors
    from berkelium_neural.render.reference import ReferenceRenderer
    from berkelium_neural.text3d.dataset import load_manifest
    from berkelium_neural.text3d.generate import generate_mesh, mesh_to_gaussians
    from berkelium_neural.text3d.geometry import clean_mesh, validate_mesh
    from berkelium_neural.text3d.train import file_sha256, load_model
    root = Path(a.dataset)
    recs = {r.asset_id: r for r in load_manifest(root, {"test"})}
    cache = ROOT / "data" / "models"
    ckpt = hf_hub_download("stabilityai/TripoSR", "model.ckpt", revision=WEIGHTS_PIN, cache_dir=cache)
    cfgp = hf_hub_download("stabilityai/TripoSR", "config.yaml", revision=WEIGHTS_PIN, cache_dir=cache)
    t0 = time.time()
    # Same weights as TSR.from_pretrained, loaded memory-mapped and assigned in place: the upstream loader holds the
    # model AND a full copy of the checkpoint (~3.4 GB anonymous memory), which a 7.8 GB host cannot always spare.
    from omegaconf import OmegaConf
    tcfg = OmegaConf.load(cfgp)
    OmegaConf.resolve(tcfg)
    model = TSR(tcfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True), assign=True)
    model.renderer.set_chunk_size(8192)
    model.eval()
    load_s = time.time() - t0
    berk = load_model(ROOT / "exports" / "textshape-base-v1" / "model.pt")
    receipt = {"milestone": "M6", "stage": "research stage C (TripoSR control assets)",
               "pins": {"code": CODE_PIN, "weights_revision": WEIGHTS_PIN, "weights_sha256": file_sha256(Path(ckpt)),
                        "license": "MIT (upstream statement for code and weights)"},
               "compatibility": {"torchmcubes": "replaced by skimage.measure.marching_cubes on CPU (shim)",
                                 "rembg": "not used (exact alpha); stub raises if called",
                                 "orientation": "tsr.utils.to_gradio_3d_orientation (TripoSR z-up -> y-up)",
                                 "transformers": __import__("transformers").__version__},
               "loader": "TSR(cfg) + torch.load(mmap=True) + load_state_dict(assign=True) (same tensors as upstream)",
               "device": "cpu", "threads": a.threads, "load_s": round(load_s, 1), "mc_resolution": a.mc_res,
               "controls": [], "environment": provenance()}
    sheets = []
    for aid in a.controls:
        rec = recs[aid]
        gt = np.load(root / rec.files["gt_mesh"])
        img, _ = control_image(rec, root)
        img.save(out / f"control_{aid}.png")
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.time()
        with torch.no_grad():
            codes = model([img], device="cpu")
            mesh = model.extract_mesh(codes, True, resolution=a.mc_res)[0]
        infer_s = time.time() - t0
        to_gradio_3d_orientation(mesh)                   # TripoSR's own z-up -> glTF/Berkelium y-up
        mesh.fix_normals()
        raw = validate_mesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64))
        # the clean-up Berkelium applies to its own marching-cubes output (geometry.clean_mesh, area < 1e-12)
        v0, f0 = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
        col0 = np.asarray(mesh.visual.vertex_colors[:, :3], np.float32) / 255.0
        v, f, removed = clean_mesh(v0, f0)
        area = 0.5 * np.linalg.norm(np.cross(v0[f0[:, 1]] - v0[f0[:, 0]], v0[f0[:, 2]] - v0[f0[:, 0]]), axis=1)
        col = col0[np.unique(f0[area >= 1e-12])]
        assert len(col) == len(v) and removed == raw.get("degenerate_faces", 0)
        import trimesh
        mesh = trimesh.Trimesh(v, f, vertex_colors=(np.clip(col, 0, 1) * 255).round().astype(np.uint8), process=False)
        rep = validate_mesh(v, f)
        mesh.export(out / f"triposr_{aid}.glb")
        sheet, tiles = orbit_sheet(v, f, col)
        Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(out / f"triposr_{aid}_orbit24.png")
        # Berkelium text-to-3D on the same asset's caption (text, not the image)
        bo = generate_mesh(berk, rec.caption, 48)
        # consumed by the Gaussian pipeline: surface Gaussians from the TripoSR mesh, splat-rendered
        vn = normalise(v) * 1.6
        sc = mesh_to_gaussians(vn, f, col, n=4000, seed=0)
        from berkelium_neural.camera import orbit_cameras
        with torch.no_grad():
            g = ReferenceRenderer().render(GaussianTensors.from_scene(sc), orbit_cameras(4, 3.2, 20.0, 128, 128)[1],
                                           RenderConfig(background=(1.0, 1.0, 1.0)))
        splat = g.rgb.numpy()
        Image.fromarray((np.clip(splat, 0, 1) * 255).astype(np.uint8)).save(out / f"triposr_{aid}_gaussians.png")
        row = {"asset_id": aid, "caption": rec.caption, "family": rec.family,
               "control_image": f"results/m6/control_{aid}.png",
               "gates": {"finite": bool(np.isfinite(v).all()), "faces": int(len(f)), "valid": bool(rep["valid"]),
                         "raw_valid": bool(raw["valid"]), "degenerate_faces_removed": removed,
                         "part_count": {"reconstructed": rep.get("components"), "ground_truth": int(gt["components"]),
                                        "correct": rep.get("components") == int(gt["components"])},
                         "components": rep.get("components"), "watertight": rep.get("watertight"),
                         "bounds": [v.min(0).round(4).tolist(), v.max(0).round(4).tolist()],
                         "units": "TripoSR canonical frame (radius 0.87); not metric",
                         "orbit24_rendered": len(tiles) == 24, "infer_extract_s_cpu": round(infer_s, 1),
                         "host_peak_rss_mib": round(max(rss0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                                                    / 1024, 1)},
               "vs_ground_truth": {
                   "triposr_from_image": compare(v, f, gt["v"], gt["f"], aligned=True),
                   "berkelium_text_to_3d_from_caption": compare(bo["vertices"], bo["faces"], gt["v"], gt["f"],
                                                                aligned=True)
                   if bo["vertices"] is not None else None,
                   "berkelium_text_to_3d_native_frame": compare(bo["vertices"], bo["faces"], gt["v"], gt["f"],
                                                                aligned=False)
                   if bo["vertices"] is not None else None},
               "gaussian_pipeline": {"n": sc.n, "splat_render": f"results/m6/triposr_{aid}_gaussians.png",
                                     "finite": bool(np.isfinite(splat).all()), "std": float(splat.std()),
                                     "validate": sc.validate()},
               "outputs": [f"results/m6/triposr_{aid}.glb", f"results/m6/triposr_{aid}_orbit24.png"]}
        receipt["controls"].append(row)
        sheets.append((aid, tiles))
        tg = row["vs_ground_truth"]
        print(f"[{aid}] {infer_s:.0f}s, {len(f)} faces; chamfer TripoSR {tg['triposr_from_image']['chamfer_l1']:.4f}"
              f" vs Berkelium {tg['berkelium_text_to_3d_from_caption']['chamfer_l1']:.4f}")
    (out / "M6_receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
    print(f"receipt: {out / 'M6_receipt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
