#!/usr/bin/env python3
"""M7: fit Gaussians to observations of a real object (Google Scanned Objects) and evaluate on held-out views.

  python scripts/fit_real_object.py --object Schleich_Lion_Action_Figure
  python scripts/fit_real_object.py --object Racoon --steps 1500 --n 12000 --res 128 --publish

Fetches the pinned scan (sha256-checked, CC-BY 4.0) into data/gso/, renders training + held-out observations,
initialises Gaussians by space carving from the training silhouettes, fits them with the Berkelium reference
renderer, and writes results/m7/<object>/report.json plus comparison images. Resumable (fit checkpoint in
data/m7/<object>/). --publish stores the fitted scene and its renders as artifacts in the neural service store
(BERKELIUM_NEURAL_HOME), where the API and Studio can retrieve them.
"""
import argparse
import json
import sys
import time

import numpy as np

from _common import ROOT, provenance


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", required=True)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--n", type=int, default=12000, help="number of Gaussians (fixed; no densification)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--train-views", type=int, default=24)
    ap.add_argument("--test-views", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    import torch

    from berkelium_neural.fitting import real_object as ro
    torch.set_num_threads(a.threads)
    work = ROOT / "data" / "m7" / a.object
    out = ROOT / "results" / "m7" / a.object
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    z = ro.fetch_gso(a.object, ROOT / "data" / "gso")
    scan = ro.load_scan(z)
    train, test = ro.view_split(a.res, a.train_views, a.test_views)
    obs_p = work / f"observations_{a.res}_{a.train_views}_{a.test_views}.npz"
    t0 = time.time()
    if obs_p.exists():
        o = np.load(obs_p)
        tr_rgb, tr_a, te_rgb, te_a = list(o["tr_rgb"]), list(o["tr_a"]), list(o["te_rgb"]), list(o["te_a"])
    else:
        tr = [ro.render_scan(scan, c) for c in train]
        te = [ro.render_scan(scan, c) for c in test]
        tr_rgb, tr_a = [x[0] for x in tr], [x[1] for x in tr]
        te_rgb, te_a = [x[0] for x in te], [x[1] for x in te]
        np.savez_compressed(obs_p, tr_rgb=np.stack(tr_rgb), tr_a=np.stack(tr_a), te_rgb=np.stack(te_rgb),
                            te_a=np.stack(te_a))
    t_obs = time.time() - t0
    print(f"[{a.object}] {scan.source['faces']} faces; observations {len(train)} train + {len(test)} held-out "
          f"at {a.res}px ({t_obs:.0f}s)")

    t0 = time.time()
    init, fitted, hist = ro.fit(train, tr_rgb, tr_a, a.n, a.steps, a.seed, checkpoint=work / f"fit_{a.n}_{a.steps}.pt")
    t_fit = time.time() - t0

    ev = {}
    renders = {}
    te_depth = [ro.render_scan(scan, c, with_depth=True)[2] for c in test]      # evaluation only
    for name, sc in (("init_space_carving", init), ("fitted", fitted)):
        rr, ra, rd = ro.render_scene(sc, test, with_depth=True)
        renders[name] = rr
        ev[name] = {"heldout": ro.image_metrics(rr, ra, te_rgb, te_a), "geometry": ro.geometry_metrics(sc, scan),
                    "heldout_depth": ro.depth_metrics(rd, ra, te_depth, te_a)}
        tr_r, tr_al = ro.render_scene(sc, train[::6])
        ev[name]["train_subset"] = {k: v for k, v in ro.image_metrics(tr_r, tr_al, tr_rgb[::6], tr_a[::6]).items()
                                    if k != "psnr_per_view"}
    payload, man = fitted.to_bytes()
    (work / "fitted.soa").write_bytes(payload)
    (work / "fitted.json").write_text(json.dumps(man, indent=2))
    (work / "fitted_3dgs.ply").write_bytes(fitted.to_3dgs_ply())

    cols = list(range(0, len(test), max(1, len(test) // 4)))[:4]
    sheet = ro.contact_sheet([[te_rgb[i] for i in cols], [renders["init_space_carving"][i] for i in cols],
                              [renders["fitted"][i] for i in cols]])
    ro.save_png(out / "heldout_gt_init_fitted.png", sheet)
    ro.save_png(out / "train_observations.png", ro.contact_sheet([tr_rgb[i:i + 6] for i in range(0, 12, 6)]))

    report = {
        "milestone": "M7", "object": a.object, "source": scan.source, "normalisation": scan.transform,
        "observations": {"kind": "renders of a photogrammetry scan of a real object (not raw photographs)",
                         "renderer": "berkelium_neural.fitting.real_object.render_scan (unlit albedo, 2x2 SSAA)",
                         "res": a.res, "train_views": len(train), "heldout_views": len(test),
                         "train_elevations_deg": [15, 40], "heldout_elevation_deg": 28,
                         "cameras": {"train": [json.loads(c.model_dump_json()) for c in train],
                                     "heldout": [json.loads(c.model_dump_json()) for c in test]}},
        "fit": {"n_gaussians": a.n, "steps": a.steps, "seed": a.seed, "renderer": "reference_cpu (gsplat_v1)",
                "loss": "L1(rgb) + 0.5 L1(alpha) on black background", "init": init.meta,
                "inputs": "training observations + cameras only (scan mesh never seen by init or fit)",
                "densification": "none (fixed count)", "history": hist},
        "evaluation": ev,
        "delta_fitted_minus_init": {k: ev["fitted"]["heldout"][k] - ev["init_space_carving"]["heldout"][k]
                                    for k in ("psnr", "ssim", "mask_iou")},
        "timing_s": {"observations": round(t_obs, 1), "fit": round(t_fit, 1), "total": round(time.time() - t_all, 1)},
        "outputs": {"scene": str((work / "fitted.soa").relative_to(ROOT)), "images": [
            str((out / "heldout_gt_init_fitted.png").relative_to(ROOT)),
            str((out / "train_observations.png").relative_to(ROOT))]},
        "environment": provenance(),
    }
    import hashlib
    report["outputs"]["scene_sha256"] = hashlib.sha256(payload).hexdigest()
    old_p = out / "report.json"
    if a.publish:
        report["published"] = publish(fitted, report, out)
    elif old_p.exists():               # re-evaluation of the same fitted scene keeps its published artifact ids
        old = json.loads(old_p.read_text())
        same = old.get("outputs", {}).get("scene_sha256") in (None, report["outputs"]["scene_sha256"])
        if old.get("published") and same:
            report["published"] = {**old["published"], "note": "published by an earlier run of this same fitted scene"}
    ro.write_json(out / "report.json", report)
    h, d = ev["fitted"]["heldout"], report["delta_fitted_minus_init"]
    print(f"[{a.object}] held-out PSNR {h['psnr']:.2f} dB (Δ vs init {d['psnr']:+.2f}), SSIM {h['ssim']:.3f}, "
          f"mask IoU {h['mask_iou']:.3f}; chamfer {ev['fitted']['geometry']['chamfer_l1']}; "
          f"report {out / 'report.json'}")
    return 0


def publish(scene, report, out):
    from berkelium_neural.contracts import ArtifactKind
    from berkelium_neural.service.api import NeuralService
    svc = NeuralService()
    payload, man = scene.to_bytes()
    try:
        g = svc.store.create(ArtifactKind.GAUSSIAN_SCENE,
                             {"scene.soa": (payload, "application/octet-stream"),
                              "scene.json": (json.dumps(man, indent=2).encode(), "application/json"),
                              "scene_3dgs.ply": (scene.to_3dgs_ply(), "application/x-ply")},
                             {"pipeline": "real_object_fit", "object": report["object"],
                              "source_zip_sha256": report["source"]["zip_sha256"], "renderer": "reference_cpu"},
                             config={"n": scene.n, "fit": {k: report["fit"][k] for k in ("steps", "seed", "loss")}},
                             validation={**scene.validate(), "heldout": report["evaluation"]["fitted"]["heldout"]},
                             labels={"units": "normalized_object", "representation": "gaussian_scene",
                                     "source": "Google Scanned Objects (CC-BY 4.0)",
                                     "not": "raw photographs; observations are renders of a real-object scan"})
        sheet = (out / "heldout_gt_init_fitted.png").read_bytes()
        img = svc.store.create(ArtifactKind.IMAGE, {"render.png": (sheet, "image/png")},
                               {"pipeline": "real_object_fit:heldout_comparison", "renderer": "reference_cpu"},
                               parents=(g.id,), labels={"representation": "image", "rows": "GT | init | fitted"})
        return {"gaussian_scene": g.id, "image": img.id, "store": str(svc.root)}
    finally:
        svc.close()


if __name__ == "__main__":
    sys.exit(main())
