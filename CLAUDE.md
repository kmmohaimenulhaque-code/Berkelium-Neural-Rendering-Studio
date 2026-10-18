# CLAUDE.md — working rules for agents in this repo

Read `README.md` (Achievements section) first, then `NEURAL_RENDERING_DECISIONS.md` and `docs/ENGINEERING_LOG.md`.

## Product target
Natural-language text → **Qwen3-8B "intent compiler" (LoRA)** → `DesignProposal` JSON (CEM requirements or
procedural Geometry IR) → **Berkelium core** (deterministic: validation L0–L6, OCCT exact STEP / Manifold mesh)
→ mesh → Gaussians → splat render in Studio. Organic, non-engineering prompts fall back to the open-world
generators (`auto`: Shap-E, then zimage-triposr), and the UI labels them "not engineering geometry".

The core lives at `kmmohaimenulhaque-code/BERKELIUM-NEURAL-RENDERING-AND-SCIENCES`, branch `physics/p1-numerical`,
pinned commit `737dc35`. It is a pinned dependency. Do not fork or copy it into this repo.

## Hard rules
1. **No heavy compute on the agent host.** Dataset building, geometry verification, training and evaluation run
   on **Kaggle** via `scripts/kaggle_gpu_runner.py` (specs in `configs/kaggle/`). The agent host only edits
   code, runs fast unit tests (< 2 min) and scores small JSON outputs.
2. **Authority rule (from the core).** The model only *proposes*. Measurements, validation and verdicts come only
   from deterministic core code. Never present model output as a measured or validated value.
3. **No fabrication.** Every number in a doc or report points to a committed file. Failed and rejected runs stay
   in the record.
4. **Pre-register before training.** Write the promotion rule (decision D-0xx in `NEURAL_RENDERING_DECISIONS.md`)
   before the run. Promotion happens only when that rule passes on held-out data.
5. **Lineage.** Pin every model, dataset, adapter and code bundle by revision plus SHA-256, with run manifests.
6. **Kaggle T4 evidence is CUDA evidence.** Never claim it as ROCm/MI300X evidence.
7. **Repo size.** Weights live on Hugging Face (`Mhaquehaque/*`) or as Kaggle Datasets/Models, never in git. The
   repo gets code, configs, JSON reports and small previews (< 5 MB per change).
8. **Secrets.** `KAGGLE_TOKEN`/`KAGGLE_KEY` and `HF_TOKEN` come from the environment only. Never ask the user to
   paste keys into chat, and never echo them.

## Hardware facts (measured)
- Kaggle T4 (16 GB, **no bf16 tensor cores**). Z-Image needed bf16 (fp16 gave NaN). Watch for fp16 overflow
  in Qwen3 too: use 4-bit NF4 weights + fp16 compute + fp32 LoRA/optimizer and check the loss for NaN.
- Kaggle sessions stop at 12 h. Training must checkpoint and resume.
- The MI300X droplet (165.245.133.213) has never been reachable from agent sessions. Do not depend on it.

## Commands
```bash
python scripts/kaggle_gpu_runner.py push  --spec configs/kaggle/<spec>.json   # bundle + launch
python scripts/kaggle_gpu_runner.py status --spec configs/kaggle/<spec>.json
python scripts/kaggle_gpu_runner.py pull  --spec configs/kaggle/<spec>.json   # SHA-256 verified
pytest services/neural/tests -q          # fast tests only
```

## Git
Branch `Berkelium-Neural-Rendering-Studio-neural-v1`. Pushing from agent sessions returns 403 unless the Claude
GitHub App has access to `enigmahacker/Berkelium-Neural-Rendering-Studio`. The fallback is one small `git bundle`
handed to the owner (Codespace: `git fetch <bundle> <branch>:tmp && git merge tmp && git push`).
