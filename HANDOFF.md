# HANDOFF: Berkelium Neural Rendering Studio

Branch: `Berkelium-Neural-Rendering-Studio-neural-v1` (pushed through `8f3ac96`)
Written 2026-10-15 at context compaction. Read this before continuing.

## 1. Repository state

- Pushed commits, oldest first: `7935cd7`, `61df495`, `15bd38f`, `c96075b` (merge of origin/main), `bf2dbff`, `6646945`, `6c20ac0`, `c4a4a1a`, `8f3ac96`.
- Uncommitted, untracked:
  - `services/neural/berkelium_neural/text3d/shap_e_backend.py`: the Shap-E open-world backend. It has not yet produced a successful generation.
  - `shap_e_model_cache/`: inspect before committing anything from it.
- Referenced but missing: `configs/environments/openworld-cpu.txt`. `shap_e_backend.available()` points to it.
- `third_party/shap-e` is an editable install in `.venv-cpu`. It is not in git.
- Last recorded backend test run: 2 failed and 23 passed, both failures in `services/backend/tests/test_providers.py` (litellm-related). Re-run before calling it green.
- Last recorded neural test run: 75 passed, 4 skipped (ROCm-only). E2E: 1 passed in about 66 s.

## 2. Results already on record

- E-R1 (renderer, MI300X, ROCm): the AMD gsplat + Berkelium HIP deterministic backward gives bitwise-identical gradients. Forward is bitwise equal to gsplat. Backward is 1.32–2.21× gsplat. Analysis is in `results/renderer/E-R1_analysis.md` §6.
- E-T2 (text→3D routing, CPU): adapter routing removes the retention regression. Chamfer on base prompts goes 0.132 → 0.090. Capsule gains are kept.
- M5 (Z-Image-Turbo image prior, pinned `f332072a…`): `blocked_environment`. The host has no GPU and 15 GB free against a roughly 33 GB snapshot. Preflight is in `results/m5/E-A1_preflight.json`.
- M6 (TripoSR on Berkelium control images, CPU): 3/3 recognisable. Two "next to" pairs merge into one component. Receipt: `results/m6/M6_receipt.json`, review: `results/m6/REVIEW.md`.
- M7 (Gaussians fitted to Google Scanned Objects, CPU): held-out PSNR 31.7 / 34.7 / 32.9 dB. Summary: `results/m7/SUMMARY.md`.
- Studio end-to-end (text → 3D → Gaussian render): `tests/e2e/test_studio_text_to_3d.py` passes.

## 3. Open-world (Shap-E) bring-up: where it stopped

Goal: an open-world pretrained generator behind a generator interface, with a fixed benchmark. The native model stays for in-vocabulary prompts.

Pins:
- Code: `openai/shap-e` @ `50131012ee11c9d2617f3886c10f000d3c7a3b43` (MIT).
- Weights: `text_cond.pt` (sha256 `e6b4fa59…`), `vector_decoder.pt` (sha256 `d7e7ebbf…`). Shap-E's downloader checks these against its own `URL_HASHES`.
- CLIP: ViT-L/14 text tower, commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. Its URL embeds the hash `b8cca3fd…`.

Status:
- The first CPU generation run exited with code 1. The log was filtered, so the traceback was hidden. The log is `/tmp/claude-1000/-workspaces-Berkelium-Neural-Rendering-Studio/2a4a1520-5582-4c13-8b2b-b268cf364e45/scratchpad/shape_first.log`.
- The CLIP download (about 890 MB) looked incomplete. `data/models/shap_e` was empty.
- The last action was a foreground download of the Shap-E and CLIP weights. The user rejected it. Nothing has been downloaded since. Resume downloads only with the user's go-ahead.

Next steps, in order:
1. Confirm the CLIP file is complete and its sha256 starts with `b8cca3fd`.
2. Run one prompt on CPU in the foreground, unfiltered, and capture the traceback.
3. Generate one prompt. Record time, peak RSS, and mesh validity.
4. Add the generator interface in `text3d/generators.py`.
5. Add router `auto`: native model if every content word is in the vocabulary, otherwise Shap-E.
6. Add a scene planner for two-object prompts.
7. Benchmark `results/text3d/E-T3_openworld_cpu.json`, capped at about 40 Shap-E generations:
   - B1: held-out GSO categories with ground truth.
   - B2: CLIP R-precision, scored by a separate evaluator model.
   - B3: retention on the procedural test subset.
   - B4: composition.
8. Studio generator selector, plus provenance on the result card. Add an E2E test for an open-world prompt.
9. Docs and notices: pin and record the CLIP and Shap-E weights.

## 4. Constraints (from the master prompt, still in force)

- Do not provision paid compute, resize billable instances, or start open-ended downloads without checking cost and resources first.
- No destructive git operations. Preserve uncommitted user changes.
- Do not push without explicit authorization. The user authorized pushing this branch; do not push other branches.
- Do not claim GPU execution when a run was on CPU.
- Do not fabricate results, hardware access, or model capabilities. Do not claim AGI or ASI.

## 5. Known limitations (not fixed)

- Adapter part-count regression on capsules: 0.53 → 0.27.
- TripoSR merges side-by-side objects.
- M7 fine geometry on thin shapes gets worse. Lion within-0.02 depth: 62% → 26%.
- M5 is blocked on this host (no GPU, disk).
- Replay transmittance optimisation is not implemented.

## 6. Environments

- Host: 2 vCPU, 7.8 GB RAM, no GPU, about 12 GB free disk at the time of writing. Memory is tight: SIGTERM observed above about 3.4 GB anonymous memory.
- `.venv-cpu`: Python 3.14.2, torch 2.14.1+cpu.
- `.venv-triposr`: transformers 4.57.6 (pinned; 5.x renames ViT keys).
- ROCm env on MI300X: torch 2.8.0+rocm7.0.2, gsplat 1.5.3. hipcc at `/opt/rocm/core-10.1/bin/hipcc`.

## 7. Key paths

- Renderer benchmark: `benchmarks/renderer/bench_renderer.py`.
- Neural service: `services/neural/berkelium_neural/service/` (`executors.py`, `api.py`).
- Routing: `services/neural/berkelium_neural/text3d/routing.py`, `evaluate.py`.
- Real-object fitting: `services/neural/berkelium_neural/fitting/real_object.py`, driver `scripts/fit_real_object.py`.
- TripoSR driver: `scripts/m6_triposr_controls.py`.
- Studio: `apps/desktop/src/scene/GeneratePanel.jsx`, `neuralApi.js`, `neuralCamera.js`.
- Docs: `NEURAL_RENDERING_DECISIONS.md` (D-017 to D-022), `NEURAL_RENDERING_ROADMAP.md`, `NEURAL_DATA_PROVENANCE.md`, `THIRD_PARTY_NOTICES.md`, `docs/ENGINEERING_LOG.md`, `docs/GPU_RUN_PLAN.md`.
