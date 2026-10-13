# Third-party components and code ownership

## Original Berkelium code (MIT, this repository)

- `services/neural/**` — contracts, camera, scene format, reference renderer, gsplat adapter, HIP extension
  (`csrc/bk_det_bwd.h`, `csrc/bk_det_bwd.hip`, `csrc/torch_bindings.cpp`, `csrc/hipcpu_capi.cpp`), text-to-3D
  grammar/dataset/model/LoRA/training/evaluation, service layer, tests.
- `benchmarks/**`, `scripts/**` (new files), `configs/**`, `apps/desktop/src/ScenePanel.jsx`, `apps/desktop/src/neuralApi.js`,
  `tests/e2e/**`.
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

## Models and data not used

No pretrained model weights and no third-party 3D assets, images or captions are used. External candidates
(Shap-E, Point-E, TripoSR — MIT; Zero123++ — CC-BY-NC-4.0, rejected as default) require separate approval
before any download.
