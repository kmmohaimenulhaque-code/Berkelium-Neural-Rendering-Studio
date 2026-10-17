# HANDOFF: Berkelium Neural Rendering Studio

Branch: `Berkelium-Neural-Rendering-Studio-neural-v1`. Written 2026-10-16 (replaces the 2026-10-15 handoff, which
predated E-T3 P0-P5 and is superseded). Evidence for every number below is in the files named next to it.

## 1. State

- Product path: Studio Scene → Generate → `auto` → native model (procedural prompts) or the first *available*
  open-world generator in E-T3 evidence order (Shap-E, then zimage-triposr; decision D-029) → mesh → Gaussians →
  splat render. Skipped generators and the host hardware are recorded in the provenance card.
- Generators: `berkelium-textshape` (CPU), `shap-e-text300m` (CPU minutes / GPU ~1 min),
  `zimage-triposr` (Z-Image-Turbo → U²-Net → TripoSR; GPU only, ~2.5 min on a T4; unavailable with the reason on CPU).
- GPU work runs on Kaggle through `scripts/kaggle_gpu_runner.py` (job specs in `configs/kaggle/`). Credentials:
  `KAGGLE_TOKEN` (or `KAGGLE_KEY`) in the environment; the account handle is read from the API.
- Tests (this host, 2026-10-16): `pytest services/neural/tests` 127 passed, 9 skipped (ROCm/GPU-only) with
  `test_durable_jobs.py` run separately; `apps/desktop` `npm test` 35 passed; ESLint clean on changed files (9
  pre-existing errors in files not touched).

## 2. Results on record (2026-10-16)

| Result | Where |
|---|---|
| E-T3 complete: Shap-E B1 F@0.05 0.669 / B2 R@1 0.95; zimage-triposr v2 0.603 / 0.85 (better B4 composition); native 0.566 / 0.05 | `results/text3d/E-T3_openworld_cpu.report.md` |
| D-029 applied: Shap-E first, zimage-triposr second | `NEURAL_RENDERING_DECISIONS.md`, `text3d/generators.py` |
| M5 pass on CUDA (Kaggle T4): bf16 (fp16 NaN), 768², ~122 s/anchor | `results/m5/E-A1_receipt.json` |
| zimage-triposr v1 (colour key) 37/80, v2 (U²-Net) 80/80 | `docs/ENGINEERING_LOG.md` (2026-10-16) |
| E-T4: capsule adapter v2 (replay + separation margin) NOT promoted (part count 0.40 < 0.53) | `results/text3d/E-T4_capsule_replay.report.md` |
| Anchor LoRA trained (bf16, 400 steps, T4) and NOT promoted (D-030: B2 R@1 0.25 vs 0.85, B1 F@0.05 0.546 vs 0.603) | `exports/zimage-anchor-lora-v1/MODEL_CARD.json` |

Older results (E-R1 on MI300X, E-T1/E-T2, M6, M7) are unchanged; see `README.md`.

## 3. Constraints (still in force)

- No paid compute or open-ended downloads without checking cost first. Kaggle T4 quota is the owner's weekly quota.
- Kaggle T4 results are CUDA evidence only, never ROCm/AMD evidence.
- No promotion of a model, adapter or routing order without its pre-registered rule passing on E-T3/E-T4.
- Do not fabricate results; failed and rejected runs stay in the record.

## 4. Open items

1. Anchor LoRA: done, not promoted. A next attempt should keep prompt fidelity (e.g. mix ordinary captioned images
   into the training set, fewer steps) and must pass D-030 again.
2. ROCm/MI300X: M5 and zimage-triposr have not run on AMD hardware (R5 in `docs/GPU_RUN_PLAN.md`).
3. Capsule part merging remains (v1 0.27; v2 0.40 not promoted).
4. U²-Net fills enclosed background (bicycle wheels); bf16 on T4 is slow (no native bf16).
5. MI300X droplet (165.245.133.213) was not reachable from the agent container (port 22 timed out); a new
   session with the IP allowed in Network access and `MI300X_SSH_KEY` loaded is needed.
6. Pushing from the agent session needs the Claude GitHub App on `enigmahacker/Berkelium-Neural-Rendering-Studio`.
