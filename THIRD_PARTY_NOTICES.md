# Third-party components and code ownership

## Original Berkelium code (MIT, this repository)

- `services/neural/**` — contracts, camera, scene format, reference renderer, gsplat adapter, HIP extension
  (`csrc/bk_det_bwd.h`, `csrc/bk_det_bwd.hip`, `csrc/torch_bindings.cpp`, `csrc/hipcpu_capi.cpp`), text-to-3D
  grammar/dataset/model/LoRA/training/evaluation, service layer, tests.
- `benchmarks/**`, `scripts/**` (new files), `configs/**`, `apps/desktop/src/neuralApi.js`,
  `apps/desktop/src/scene/GeneratePanel.jsx`, `apps/desktop/src/scene/neuralCamera.js`, `tests/e2e/**`.
- Procedural dataset `berkelium-procedural-shapes-v1` and the model weights trained on it.

No source file from an upstream project is copied into this repository.

## Upstream-derived algorithmic content

- `services/neural/csrc/bk_det_bwd.hip`: the per-pixel gradient expressions (alpha/transmittance recurrence,
  conic and mean2d derivatives, 1/255 alpha threshold, 0.999 alpha cap, 1e-4 exclusive termination) follow
  `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` of AMD gsplat v1.5.3b2 so that the two produce numerically matching
  gradients. The accumulation scheme (per-intersection slots, fixed-shape block reductions, ordered per-Gaussian
  segment sums, no atomics) is Berkelium's. gsplat: Apache License 2.0 (upstream Nerfstudio portions) and MIT
  (AMD modifications), Copyright (C) 2024-2026 Advanced Micro Devices, Inc. and the Nerfstudio team.
- `render/reference.py` `gsplat_v1` profile reproduces the same published rules; projection parity is tested against
  gsplat's own pure-PyTorch reference (`gsplat/cuda/_torch_impl.py`), which is imported from the pinned checkout
  at test time, not copied.

## Pinned dependencies fetched by setup scripts (not vendored)

| Component | Pin | License | Use |
|---|---|---|---|
| AMD-Ecosystem/gsplat | tag `v1.5.3b2`, commit `ff8bc5489545bcc1b5a2b997d975834429a0d7af` | Apache-2.0 + MIT (AMD), see its `LICENSE` and `NOTICE.txt` | differentiable rendering foundation (ROCm) |
| ROCm/HIP-CPU | commit `e112c935057434897bb12d9ab3910380a8bd5f58` | MIT | CPU execution of the Berkelium HIP source for validation |
| oneTBB (system package) | distribution version | Apache-2.0 | HIP-CPU runtime dependency |
| three.js | `^0.186` (npm) | MIT | Studio viewport |
| PyTorch, NumPy, SciPy, scikit-image, trimesh, Pillow, FastAPI, pydantic | `configs/environments/*.txt` | BSD / MIT / Apache-2.0 | runtime |

## Models and data used (fetched at pinned versions, not vendored)

| Component | Pin | License | Use |
|---|---|---|---|
| TripoSR code (VAST-AI-Research) | commit `107cefdc244c39106fa830359024f6a2f1c78871` (`third_party/TripoSR`) | MIT | M6 single-image reconstruction (model code imported, not copied) |
| TripoSR weights (stabilityai/TripoSR) | revision `5b521936b01fbe1890f6f9baed0254ab6351c04a`, `model.ckpt` sha256 `429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee` | MIT (upstream statement) | M6 |
| DINO ViT-B/16 config (facebook/dino-vitb16) | `config.json` fetched by TripoSR at load | Apache-2.0 | M6 (architecture config only; weights are inside the TripoSR checkpoint) |
| Google Scanned Objects (Google LLC) | per-object Fuel zip sha256 in `services/neural/berkelium_neural/fitting/real_object.py` | CC-BY 4.0 | M7 real-object observations and evaluation |

Attribution (CC-BY 4.0): "Google Scanned Objects" by Google LLC, models `Schleich_Lion_Action_Figure`, `Racoon`,
`Room_Essentials_Bowl_Turquiose` from https://app.gazebosim.org/GoogleResearch — Downey et al., "Google Scanned
Objects: A High-Quality Dataset of 3D Scanned Household Items", ICRA 2022. Berkelium renders the scans to images and
fits Gaussians to those images; the derived images and fitted scenes in `results/m7/` are adaptations and carry
the same attribution. The scans themselves are not redistributed (fetched into `data/gso/`, git-ignored).

## Models not used

Z-Image-Turbo (`f332072a`, Apache-2.0 per its card) is wired for the M5 smoke test but has not been downloaded or
run (no GPU on the development host). Zero123++ (CC-BY-NC-4.0) remains rejected as a default. Shap-E and Point-E
(MIT) are not used.
