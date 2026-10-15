# M7 — Gaussian fitting to real-object observations (2026-10-15)

Data: Google Scanned Objects (Google LLC, CC-BY 4.0), pinned by zip sha256. Observations: 24 training + 8 held-out
views at 128 px rendered from the textured scans (not photographs; exact cameras; held-out cameras on a separate
elevation and azimuths). Fit: 8,000 Gaussians, 1,000 Adam steps, Berkelium reference renderer (gsplat_v1), loss
L1(rgb) + 0.5 L1(alpha); initialisation by space carving from the training silhouettes only; the scan mesh is never
seen by the initialisation or the fit (evaluation only). CPU, 2 threads.

## Appearance on held-out views

| object (scan faces) | PSNR init → fit | SSIM init → fit | mask IoU init → fit | train-view PSNR (fit) | fit time (1,000 steps) |
|---|---|---|---|---|---|
| Room_Essentials_Bowl_Turquiose (5,112) | 21.02 → **31.72** dB | 0.747 → 0.954 | 0.924 → 0.988 | 31.80 dB | 6.0 min |
| Schleich_Lion_Action_Figure (15,058) | 22.20 → **34.65** dB | 0.844 → 0.981 | 0.794 → 0.970 | 35.76 dB | 5.0 min |
| Racoon (70,952) | 22.95 → **32.91** dB | 0.809 → 0.970 | 0.860 → 0.979 | 33.49 dB | 5.1 min |

## Geometry (scan used for evaluation only; units: fraction of the 1.6 normalised extent)

Rendered depth vs the scan's depth on held-out views (pixels fully covered by the scan and opaque in the render):

| object | mean abs init → fit | median init → fit | within 0.02 init → fit | within 0.05 init → fit |
|---|---|---|---|---|
| Room_Essentials_Bowl_Turquiose | 0.404 → 0.195 | 0.259 → 0.058 | 7% → 12% | 16% → **45%** |
| Schleich_Lion_Action_Figure | 0.030 → 0.027 | 0.014 → 0.026 | 62% → 26% | 87% → **97%** |
| Racoon | 0.037 → 0.023 | 0.023 → 0.022 | 44% → 40% | 76% → **99%** |

Gaussian centres (opacity ≥ 0.25) vs points on the scan surface:

| object | Chamfer-L1 init → fit | F@0.02 init → fit | F@0.05 init → fit |
|---|---|---|---|
| Room_Essentials_Bowl_Turquiose | 0.0695 → 0.0365 | 0.317 → 0.282 | 0.707 → 0.882 |
| Schleich_Lion_Action_Figure | 0.0144 → 0.0380 | 0.800 → 0.048 | 1.000 → 0.853 |
| Racoon | 0.0195 → 0.0284 | 0.722 → 0.280 | 0.950 → 0.933 |

## Reading

- Appearance: the fit adds 10–12.5 dB held-out PSNR over the initialisation; held-out PSNR is within 1.1 dB of
  training-view PSNR, so it generalises to the unseen cameras of this setup.
- Coarse geometry improves: rendered depth within 0.05 rises to 97–99% on the lion and raccoon. The bowl's concave
  interior is only partly recovered (45%): space carving cannot carve concavities and 1,000 steps do not finish it.
- Fine geometry does not improve and can get worse: on the thin lion, depth within 0.02 falls 62% → 26% and Gaussian
  centres drift off the surface (centre F@0.02 0.80 → 0.05) while the images improve. A photometric loss alone does
  not pin centres to the surface; the silhouette hull is already tight for thin convex shapes. Candidate remedies
  (not done): depth/normal regularisers, opacity and scale regularisation, more steps with densification.

Per object: `report.json` (cameras, per-view PSNR, loss history, provenance, published artifact ids),
`heldout_gt_init_fitted.png` (rows: observation | space-carving init | fitted, on held-out cameras),
`train_observations.png`. Fitted scenes are in `data/m7/<object>/` (`fitted.soa`, `fitted_3dgs.ply`) and were
published as `gaussian_scene` artifacts in the neural store; any of them renders from any camera through
`POST /v1/neural/jobs {"kind": "render", ...}`.

Reproduce: `python scripts/fit_real_object.py --object <name> --n 8000 --steps 1000 --res 128 [--publish]`
(resumes from `data/m7/<name>/fit_8000_1000.pt`; observations cached).

Limits: observations are renders of scans (no photographic noise, exposure or pose error); fixed Gaussian count
(no densification); 128 px; three objects.
