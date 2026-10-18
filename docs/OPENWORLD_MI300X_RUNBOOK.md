# Open-world text-to-3D on the MI300X: runbook (E-T3 follow-up)

> **To resume E-T3 on a GPU, use `docs/E-T3_GPU_RESUME.md`** (smoke gate, pending-unit partition, MI300X and
> Kaggle commands). This file keeps the background and the later experiments.

Status: **UNVERIFIED on any GPU.** Everything below was run only on the 2-vCPU development host (CPU). Shap-E is
pure PyTorch (no custom CUDA kernels in the text300M + STF-mesh path used here: transformer, Karras sampler, MLP
SDF/texture queries, Shap-E's own torch marching cubes; PyTorch3D is optional and absent), so it is *likely* to run
on ROCm PyTorch. That is a hypothesis, not evidence, until step 3 passes on the MI300X.

## 0. What the CPU run established (results/text3d/E-T3/P2_COST_MODEL.md, E-T3_openworld_cpu.*)

- 16 Karras steps: 827 s wall per generation on 2 vCPU (47.7 s/step), peak RSS 2.6 GB, recognisable chair.
- The CPU budget allowed 28 Shap-E generations; omitted from E-T3: the B4 Shap-E-planner arm (16 generations) and
  B1 seed-1/2 variance (24 generations).

## 1. Environment (the ROCm env used for E-R1: torch 2.8.0+rocm7.0.2, Python 3.12)

```bash
cd ~/Berkelium-Neural-Rendering-Studio && git pull
PYTHON=$(which python) bash scripts/setup_openworld.sh      # clip + shap-e (--no-deps) into that env; ~3.1 GB
                                                            # of weights into data/models/shap_e, sha256-checked;
                                                            # extracts CLIP's text tower (no TorchScript)
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"   # hip != None
```

`scripts/setup_openworld.sh` never installs or replaces torch. It refuses to continue on a hash mismatch.

## 2. Smoke test (one generation; GPU allocation ~1 min)

```bash
python scripts/openworld_generate.py "a wooden chair with four legs" --steps 16 --device cuda \
  --out results/text3d/E-T3-gpu/smoke --rss-abort-mib 60000
```

Pass: exit 0, `report.json` `valid: true`, `identity.device == "cuda"`, `preview.png` shows a chair. Record
`sample_s`, `decode_s`, `torch.cuda.max_memory_allocated()` (add it to the report if missing). On failure, keep the
full traceback (`run.log`). Do not retry blindly. Likely fault points: `torch.load(mmap=True)` (CPU only here; on GPU
the state dict is loaded to CPU, then copied), fp16 autocast in `sample_latents(use_fp16=True)`, and Shap-E's
torch marching cubes on device.

## 3. Equivalence check before any benchmark (CPU vs GPU)

Same prompt and seed with `--device cuda --fp32`, compared with the CPU mesh in
`results/text3d/E-T3/units/shap-e-text300m/` (or the P2 chair) by `openworld_eval.align_and_score`. Expect Chamfer-L1 < 0.01 in unit extent. fp32 sampling
on different hardware is not bitwise equal, but the shape must match. Only then enable fp16.

## 4. The single next measurable experiment: E-T3-GPU

Rerun E-T3 at the full design on the MI300X, without changing the frozen prompt list or any scoring rule:

1. Same config, `--device cuda` (add `"device": "cuda"` to `shap_e` in a copy of
   `benchmarks/configs/text3d_openworld_cpu_v1.json` named `..._gpu_v1.json`, with `max_shap_e_generations: 69`).
2. Run the omitted units: the B4 Shap-E-planner arm and B1 seeds 1-2 (variance). Plus a steps sweep on B1:
   steps ∈ {16, 32, 64} × guidance ∈ {7.5, 15}.
3. Report the same tables as `E-T3_openworld_cpu.report.md`, plus the per-setting time and VRAM.

Decision rule (written before running): adopt a larger step count in the product default only if B1 mean F@0.05
improves by ≥ 0.05 over 16 steps, with no drop in B2 R@1, at ≤ 60 s per generation on the MI300X.

## 5. After E-T3-GPU: the hybrid image-prior backend (D-006)

`zimage-triposr` is registered but GPU-gated (it never emits output). After E-T3-GPU, run M5 (Z-Image-Turbo,
`scripts/m5_image_prior_smoke.py --i-approve-gpu-cost`). Then implement `ZImageTripoSR.generate` (text →
Z-Image → TripoSR, the M6 code path) and score it on the same E-T3 B1/B2. Keep it only if it beats Shap-E on
B1 F@0.05 or B2 R@1.

## Cost guard

Each step above is bounded: setup about 10 min (downloads), smoke about 2 min, equivalence about 5 min, E-T3-GPU at
an assumed ≤ 60 s per generation is under 2 h. Stop and report if any step exceeds 2x its estimate.
