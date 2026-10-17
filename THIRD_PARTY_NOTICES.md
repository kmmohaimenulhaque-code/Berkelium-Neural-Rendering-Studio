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
| Google Scanned Objects (Google LLC) | per-object Fuel zip sha256 in `services/neural/berkelium_neural/fitting/real_object.py` (M7) and `benchmarks/configs/text3d_openworld_cpu_v1.prompts.json` (E-T3) | CC-BY 4.0 | M7 real-object observations and evaluation; E-T3 B1 ground truth |
| Shap-E code (openai/shap-e) | commit `50131012ee11c9d2617f3886c10f000d3c7a3b43` (`third_party/shap-e`, installed editable, model code imported, not copied) | MIT | E-T3 open-world text-to-3D generator (`text3d/shap_e_backend.py`) |
| Shap-E weights (OpenAI) | `text_cond.pt` sha256 `e6b4fa599a7b3c3b16c222d5f5fe56f9db9289ff0b6575fbe5c11bc97106aad4`, `vector_decoder.pt` sha256 `d7e7ebbfe3780499ae89b2da5e7c1354012dba5a6abfe295bed42f25c3be1b98`, configs pinned by Shap-E's `URL_HASHES` | MIT (released with the code) | E-T3 |
| CLIP code (openai/CLIP) | commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` (pip, `configs/environments/openworld-cpu.txt`) | MIT | text encoder for Shap-E; evaluator for E-T3 B2 |
| CLIP ViT-L/14 weights (OpenAI) | `ViT-L-14.pt` sha256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`; text tower extracted to `clip_vit_l14_text.pt` (derived, records the source sha256) | MIT | Shap-E text conditioning (text tower only) |
| CLIP ViT-B/32 weights (OpenAI) | `ViT-B-32.pt` sha256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` | MIT | E-T3 B2 evaluator only (not used for generation) |
| Z-Image-Turbo (Tongyi-MAI) | revision `f332072aa78be7aecdf3ee76d5c247082da564a6`; every file pinned by the hash the Hugging Face API publishes (sha256 for the 7 safetensors shards, git blob sha1 for configs/tokenizer), listed in `text3d/zimage_triposr.py` `ZIMAGE` | Apache-2.0 (model card metadata at the pinned revision) | zimage-triposr anchor images (M5, E-T3); weights cast to fp16 at load, never redistributed |
| Qwen3 text encoder (inside the Z-Image-Turbo snapshot) | same revision, `text_encoder/model-0000{1,2,3}-of-00003.safetensors` sha256 in `ZIMAGE` | Apache-2.0 (shipped in the Apache-2.0 snapshot) | anchor prompt embeddings |
| TripoSR (zimage-triposr stage 4) | code `107cefdc…` and weights `5b521936…` as above (M6) | MIT | reconstruction of keyed anchors on CUDA (Kaggle T4) |
| diffusers / transformers / accelerate / peft | `configs/environments/zimage-cuda.txt`, `triposr-cuda.txt` | Apache-2.0 | Z-Image pipeline, LoRA training, TripoSR model code |
| Kaggle CLI | `kaggle==2.2.4` (`configs/environments/kaggle-control.txt`) | Apache-2.0 | control plane for `scripts/kaggle_gpu_runner.py` |

Attribution (CC-BY 4.0): "Google Scanned Objects" by Google LLC, models `Schleich_Lion_Action_Figure`, `Racoon`,
`Room_Essentials_Bowl_Turquiose` from https://app.gazebosim.org/GoogleResearch — Downey et al., "Google Scanned
Objects: A High-Quality Dataset of 3D Scanned Household Items", ICRA 2022. Berkelium renders the scans to images and
fits Gaussians to those images; the derived images and fitted scenes in `results/m7/` are adaptations and carry
the same attribution. The scans themselves are not redistributed (fetched into `data/gso/`, git-ignored).

E-T3 additionally uses these Google Scanned Objects (same licence and citation) as B1 ground truth:
`Cole_Hardware_Mug_Classic_Blue`, `Threshold_Porcelain_Teapot_White`, `Threshold_Porcelain_Pitcher_White`,
`Cole_Hardware_Deep_Bowl_Good_Earth_1075`, `Spritz_Easter_Basket_Plastic_Teal`,
`Timberland_Mens_Earthkeepers_Newmarket_6Inch_Cupsole_Boot`, `CLIMACOOL_BOAT_BREEZE_IE6CyqSaDwN`, `SCHOOL_BUS`,
`TURBOPROP_AIRPLANE_WITH_PILOT`, `Lovable_Huggable_Cuddly_Boutique_Teddy_Bear_Beige`, `Pony_C_Clamp_1440`,
`Elephant`. The scans are only rendered for the evaluator's reference row and compared against generated meshes;
the renders in `results/text3d/E-T3/` are adaptations with the same attribution.

Shap-E: Jun & Nichol, "Shap-E: Generating Conditional 3D Implicit Functions", 2023 (MIT). CLIP: Radford et al.,
"Learning Transferable Visual Models From Natural Language Supervision", 2021 (MIT). Weights are fetched by
`scripts/setup_openworld.sh` into `data/models/` (git-ignored) and never committed. Meshes generated by Shap-E are
labelled "model-generated, normalized units, not measured geometry".

Anchor-LoRA training set (`configs/datasets/anchor_lora_gso_v1.json`): 36 further Google Scanned Objects (same
licence and citation, zip sha256 pinned in that file), rendered to images and used to train the Z-Image anchor LoRA
(`exports/zimage-anchor-lora-v1`, when present). The renders and the adapter are adaptations of CC-BY 4.0 material and
carry this attribution; no scan is redistributed. None of these objects is in an E-T3 B1/B2 category.

## Models not used

Z-Image-Turbo is used since 2026-10-16 (above), on Kaggle T4 GPUs only. Zero123++ (CC-BY-NC-4.0) remains rejected as a default. Point-E (MIT) is not
used. Shap-E is used since E-T3 (above).
