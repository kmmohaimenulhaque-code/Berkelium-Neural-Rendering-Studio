# Engineering log — Berkelium neural slice

Concise, dated, evidence-first. Newest last.

## 2026-10-13 — Audit (baseline: uploaded `Berkelium-Neural-Rendering-Studio-main.zip`, sha256 `b3dba1f6…ef6d`)

Repository reality (no `.git` in the archive; a local baseline commit was created from it):

| Component | Classification | Note |
|---|---|---|
| Electron main/preload | implemented | contextIsolation true, nodeIntegration false, narrow preload (file open/save). Docs NR-0.2 said otherwise; code is now correct. `sandbox` not set. |
| React shell, chat (Fireworks), Monaco code panel | implemented | renderer calls FastAPI directly at 127.0.0.1:8000 |
| 3D viewport | placeholder → **implemented now** | was "Three.js viewport (Day 2)" |
| `packages/*` (18 JS modules) | simulated / dead code | `Math.random()` fidelity, `setTimeout` jobs; not imported by the app; no package.json per package |
| FastAPI backend | implemented | health, chat, OCR, PDF; no neural routes before this work |
| Tests | missing → **added** | README claimed a `tests/` dir that did not exist |
| Root `package.json` | missing | README lists one |
| `services/backend/venv/` | committed macOS Python 3.9 virtualenv | should be removed from version control (not deleted here; owner decision) |
| Neural rendering / text-to-3D | documents only → **implemented now** | 8 design docs (NR-0.2) were explicitly proposals |

AMD gsplat: the ROCm port exists only on `release/1.5.3b2` = tag `v1.5.3b2` = `ff8bc5489545bcc1b5a2b997d975834429a0d7af`;
`main` and `amd-main` equal upstream nerfstudio `512d366`. Support: ROCm 6.4.3/7.0.0, PyTorch 2.6/2.8, Python
3.10/3.12, MI300X. Licence Apache-2.0 (upstream) + MIT (AMD), `NOTICE.txt`. Backward rasterization accumulates
cross-tile gradients with float `atomicAdd` (wave64 rocprim reductions within a tile) → summation order across tiles
is scheduling-dependent. The historical MI300X environment (ROCm 7.0, PyTorch 2.10) is outside AMD's gsplat matrix.

## 2026-10-13 — Renderer milestone (CPU evidence)

- Reference renderer (`gsplat_v1` + `berkelium_v02` profiles): analytic fixtures (single Gaussian centre value,
  two-layer compositing, exclusive-vs-inclusive termination), projection parity with gsplat's own pure-PyTorch
  `_fully_fused_projection` (atol 1e-9 on means2d, rtol 1e-9 conics), FP64 central-difference gradcheck on all 5
  parameter families — pass.
- Berkelium HIP kernel `bk_det_bwd.hip` executed unmodified under HIP-CPU (warpSize 64): gradients vs FP64 autograd
  within 2e-5 + 2e-3·|ref| on 3 image sizes incl. partial tiles; bitwise-identical gradients over 5 runs with
  parallel block scheduling; full-renderer world-space gradients match the reference.
- Independent review (separate agent, code only): one HIGH defect — `torch_bindings.cpp` would not compile on ROCm
  (2-arg macro called with 3 args). Fixed; added `test_binding_hipify.py` (PyTorch hipify + syntax check), which
  reproduces the original error on the old code. Also fixed: device guard, shape/device checks, build from a private
  copy (hipify writes next to sources), dropped `-ffp-contract=off` to match gsplat's flags, ROCm path of
  `BerkeliumRenderer` (host-side tile intersection). Added the tile-level `bin_final` skip (gsplat's optimisation,
  per block, deterministic).
- NOT verified: any GPU execution (no ROCm device here). E-R1 is defined and refused until approved.

## 2026-10-13 — Text-to-3D milestone

- Dataset `berkelium-procedural-shapes-v1`: 1,422 assets, 78 families; family-level splits (train 43, val 4, test 8,
  adapter_train 13, adapter_test 10 families); zero family/asset overlap (checked); GT meshes (64³ marching cubes of
  exact analytic SDFs), 4 eval views + 8 pilot train views per family; manifest sha256 `5d35e50e…`.
- Findings while building: marching cubes emits zero-area faces at cone apexes (now cleaned and counted); a cone
  stacked on a cone apex touches at a point (2 components) — the part-count rubric therefore uses the GT mesh's
  component count, not the relation word.
- Model BerkeliumTextShape v1 (362k params) trained from scratch on CPU (4,000 steps, 50 min wall, 2 cores, 0 GPU-h,
  1.46 MB, sha256 `4989d90b…`). LoRA r=8 + token-embedding delta for the held-out concept `capsule` (28,688 trainable
  params, 1,500 steps, 22 min, 122 KB, sha256 `fa5b2eae…`). Training resume is bitwise-identical to an uninterrupted run.
- E-T1 (`results/text3d/E-T1_cpu.report.md`, paired bootstrap 95% CIs over assets):
  - base vs retrieval baseline, held-out structure families (n=32): Chamfer-L1 0.124 → 0.090 (Δ −0.035, CI
    [−0.052, −0.018]); F@0.05 0.30 → 0.40; held-out-view PSNR 11.9 → 13.7 dB (CI [+1.0, +2.6]); colour accuracy
    0.47 → 0.94; **part-count accuracy regressed 0.69 → 0.47** (model merges separated parts).
  - LoRA vs pretrained base on unseen-concept families (n=30): Chamfer 0.134 → 0.081 (CI [−0.071, −0.036]); F@0.05
    0.25 → 0.43; PSNR +2.2 dB (CI [+1.5, +3.0]); normal consistency +0.12; **part count regressed 0.53 → 0.27**.
  - **Retention regression**: the adapter applied to base prompts worsens them (Chamfer +0.042, CI [+0.029, +0.055]).
    Next: adapter training with base-data replay, or route the adapter only when its concept appears.
  - LPIPS not_run (needs pretrained network weights). Geometry is blobby at this model size (see E2E screenshot).
- Bug found by the benchmark (failed run preserved in `results/text3d/E-T1_cpu.20261013T154829Z.json`): in eval mode
  `nn.TransformerEncoderLayer` uses a fused fast path that reads `linear.weight` directly, bypassing LoRA. Fixed by
  exposing merged LoRA weights; regression test asserts fast path == module path and adapter != base.

## 2026-10-13 — Integration

- `/v1/neural` mounted on the existing backend behind the same API key; validation before side effects (422/503/429
  tested); jobs persisted in SQLite with D-010 states, cancellation, restart → `resumed`.
- Studio Scene workspace renders the generated GLB and the Gaussian-scene centres from the immutable artifacts;
  E2E (Playwright + SwiftShader WebGL) asserts the browser-loaded face count equals the backend's validated report.

## 2026-10-13 — Renderer benchmark E-R0 (CPU; `results/renderer/E-R0_cpu.report.md`)

- Spec smooth fixture (N=16, 32²): forward RGB/alpha/depth and all 5 gradient families within spec tolerance —
  **pass**; gradients bitwise repeatable (5 runs) — **pass**; FP64 FD gradcheck of the oracle — **pass**.
- Agreement on larger scenes (not gates): 0 entries beyond spec tolerance at N=200/1000; at N=3000 (470 pairs/tile)
  19/9000 world-mean entries exceed it (max 9.8e-3 vs 3.9e-4 for FP32 autograd).
- Diagnosis (`benchmarks/renderer/precision_diagnosis.py`, `results/renderer/E-R0_precision.json`): an FP64 copy of
  the same kernel matches the FP64 oracle to ≤1e-12 → logic exact; FP32 division-based transmittance recovery (the
  gsplat algorithm the kernel reproduces for parity) has 7–30× the FP32 error of product-form autograd. This is
  shared with gsplat by construction; E-R1 measures gsplat against the same oracle. Planned Berkelium improvement:
  replay/checkpoint transmittance recovery (spec §8), measured against gsplat.
- Fixed-budget fit: held-out PSNR 21.65 (reference) vs 21.64 dB (Berkelium kernel).
- Performance of the HIP kernel is not measurable here (HIP-CPU emulation) → `not_applicable`; gsplat → `not_run`.

## 2026-10-14 — MI300X E-R1 recorded; repair pass (CPU only, no GPU job)

- E-R1 ran on one MI300X VF (ROCm 7.0, PyTorch 2.8.0+rocm7.0.2): the Berkelium HIP extension built with hipcc and ran;
  the small-case acceptance gate passed 16/16 for both backends; Berkelium gradients bitwise repeatable in all cases
  (gsplat 3–10 unique hashes / 10 runs); Berkelium backward 1.43–2.29× gsplat's. Details: `results/renderer/E-R1_analysis.md`.
- Correction: forward oracle rows on non-gate cases were reported as acceptance failures (100k RGB 1.0456e-3 > 2e-4).
  The protocol makes the FP64 oracle a gate on small cases only, so these rows are now diagnostic (`passed: null`,
  raw value + tolerance + `within_tolerance` kept). Applied by `benchmarks/renderer/classify.py` to new runs and, without
  modifying them, to recorded results.
- CPU diagnosis: an independent FP32 implementation reproduces the 100k error (RGB 1.0450e-3, alpha identical to the
  GPU value). The cause is FP32 projection pushing one Gaussian across the 1/255 alpha threshold at the worst pixel,
  not compositing arithmetic or depth order.
- Added: direct backend-vs-baseline forward/gradient comparison per case (E-R1: `not_run`, needs a GPU rerun); runtime
  authorization record (static `approved` unchanged; CLI flag recorded separately); detach before scalar conversion.
- Tests fixed for GPU machines: hipify shims (PyTorch 2.8 ROCm stream spelling), simulated no-HIP-device, simulated
  gsplat unavailability split from a real gsplat success test (hip-marked).

## 2026-10-15 — Integration sprint: Studio product path, M5–M8, MI300X rerun

Development host: GitHub Codespace, 2 vCPU, 7.8 GB RAM, no GPU, Python 3.14.2, torch 2.14.1+cpu.

**Found and fixed before anything else**
- The committed backend (`services/backend/main.py`) never mounted `/v1/neural`, and the committed Studio never
  mounted the neural panel or its CSS (earlier uploads lost them); the model registry pointed at `runs/`, which does
  not exist in a clone. A fresh checkout therefore could not generate anything from Studio.
- `origin/main` had 14 commits of Studio work (scene editor, settings, multi-provider backend). Merged into this
  branch (merge commit, no history rewrite); conflicts resolved keeping both sides.

**Product path (P0) — verified**
- Studio Scene → *Generate*: prompt → `text_to_3d` job (released `exports/textshape-base-v1`, adapter routing) →
  mesh imported into the scene through main's import path → Gaussian scene → splat render by the reference renderer
  (GPU path when gsplat is usable) shown in Studio → *Render this view*: the three.js camera + model transform →
  OpenCV `CameraSpec` (`scene/neuralCamera.js`, 5 unit tests incl. exact agreement with three.js projection) →
  `render` job → image. Optional Gaussian refinement (100/300 steps) from the panel.
- `tests/e2e/test_studio_text_to_3d.py` (real backend process, built UI, headless Chromium/SwiftShader, released
  checkpoint): **passed in 140 s** — 6,780-face mesh in the scene (Model panel shows the same triangle count and the
  artifact id), 4,000-Gaussian orbit render, view render 384×357 with all 4,000 Gaussians in view, backend restart →
  job still `completed`, reopened from *Recent*. Headless: `scripts/start_backend.sh` +
  `scripts/demo_text_to_3d.py` verified ("a green capsule next to a white sphere" → routed to the capsule adapter,
  10,008-face watertight mesh, 4,000 Gaussians, splat render).
- Status-bar health check used a hard-coded port; now uses the configured backend URL.

**M8 durable jobs — pass**
- Bug found by reading the executor: a job resumed after a crash regenerated its mesh; the mesh record contains a
  timing field, so it got a new artifact id and the job listed two meshes. Fixed: the mesh stage is checkpointed in
  the job workdir *before* the job lists the artifact; resumed jobs reuse it. Fit checkpoints keep `psnr_before`.
  Job reads report `missing_artifacts`.
- `test_durable_jobs.py` on a real uvicorn process: SIGKILL after the first fit checkpoint → restart → job resumed
  (first post-restart checkpoint at step 50, not 25) → exactly one mesh/Gaussian/image artifact, same mesh id →
  clean restart → still retrievable; a deleted record is reported and 404s. **1 passed (386 s).**

**M5 image prior — blocked_environment (not run)**
- `scripts/m5_image_prior_smoke.py` (pinned Z-Image-Turbo `f332072a`, ≤4 anchors, 9 steps, CFG 0, repeatability,
  `geometry_generated: false`). Preflight here: no GPU, diffusers absent, 14.8 GB free vs ~33 GB snapshot →
  `results/m5/E-A1_preflight.json`. Requires R5 on an MI300X.

**Renderer (P6)**
- MI300X rerun recorded (`results/renderer/E-R1_analysis.md` §6): direct comparison now measured — forward bitwise
  equal to gsplat; gradients within 4.1e-7 relative; Berkelium bit-identical over 10 repeats; backward 1.32–2.21×.
- Replay-based transmittance recovery: **not implemented**. Exact replay needs per-element forward transmittance
  during the back-to-front sweep (O(m²/256) recomputation per tile, or per-pixel-per-batch scratch), a structural
  change to a kernel that is verified on MI300X; its accuracy is checkable on HIP-CPU but its purpose (performance at
  equal determinism) is only measurable on a GPU, which this host lacks. Next step: implement as an opt-in kernel
  variant behind the existing interface, accept only if HIP-CPU error vs the FP64 oracle drops and MI300X E-R1 timing
  stays within budget.

**P5 text-to-3D adapter — E-T2 (`results/text3d/E-T2_routing_cpu.report.md`), CPU, 24 min**
- Same protocol, dataset and splits as E-T1 (evaluation splits rebuilt with `prepare_dataset.py --splits test
  adapter_test`; all five split hashes equal the released model card's), released checkpoints from `exports/`.
- Routing (`adapter=auto`, the Studio default) sent 32/32 base-test prompts to the base model and 30/30 capsule
  prompts to the adapter. Routed outputs are per-asset identical to base on base prompts and to the adapter on capsule
  prompts (paired deltas exactly 0).
- vs always-on adapter, base prompts: Chamfer-L1 0.132 → 0.090 (paired Δ −0.042, 95% CI [−0.055, −0.029]),
  F@0.05 0.25 → 0.40, view PSNR +1.6 dB (CI [1.06, 2.19]), part count 0.28 → 0.47: the retention regression is gone.
- vs base, capsule prompts: Chamfer 0.134 → 0.081 (Δ −0.053, CI [−0.071, −0.036]), PSNR +2.2 dB, normal consistency
  +0.12 — the adapter's gains are kept.
- Not fixed: the adapter's **part-count regression on capsule prompts (0.53 → 0.27)** — it merges separated parts;
  routing cannot change what the adapter generates. Needs adapter retraining (e.g. part-separation-aware loss or
  base-data replay); not attempted in this sprint (no evidence yet that a CPU retrain would fix it).
- Base model on `test` reproduces E-T1 (Chamfer 0.0899 vs 0.090 recorded).

**M6 TripoSR control assets — engineering findings**
- Pinned code `107cefd` + weights `5b521936` (sha256 `429e2c6b…`) run on CPU after three compatibility fixes, none
  touching model code or weights: (1) `torchmcubes` (CUDA extension) → scikit-image marching cubes behind a shim with
  torchmcubes' axis convention (unit-tested against TripoSR's `[2, 1, 0]` reorder); (2) transformers 5 renamed the
  ViT parameters stored in the checkpoint (`encoder.layer.N.attention.attention.query` → `layers.N.attention.q_proj`),
  so loading failed with missing/unexpected keys → separate env pinned to transformers 4.57.6; (3) the upstream loader
  holds the model and a full checkpoint copy (~3.4 GB anonymous memory); on this 7.8 GB host the process was sent
  SIGTERM ~19 s after start (no kernel OOM event; host-level memory pressure) → weights are loaded memory-mapped and
  assigned in place (same tensors); peak RSS 3.5 → 2.7 GiB and the run completes.
- TripoSR's output is z-up; converted with its own `to_gradio_3d_orientation`. Reconstructions are tilted by the
  control camera's 16° elevation (TripoSR assumes a level camera) — handled by the evaluation's similarity alignment.
- Marching cubes emits zero-area faces (1 on two of the meshes), which Berkelium's validator rejects; the TripoSR
  output now gets Berkelium's own `clean_mesh` (trimesh's `nondegenerate_faces` did not catch them).
