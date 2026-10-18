# E-R1 analysis (MI300X, 2026-10-13): oracle-role correction and forward discrepancy

Sources: `E-R1_mi300x.json` (sha256 `da17f2b2…52cb3`, byte-identical to the history copy
`E-R1_mi300x.20261013T200512Z.json`; both unchanged by this analysis), the reclassified comparison
`E-R1_mi300x.comparison.json` / `.report.md` / `.csv` (generated from the recorded JSON; no measurement added or changed),
and the CPU diagnosis `E-R1_forward_diagnosis_cpu.json`. No GPU job was run for this analysis.

## 1. Classification (was: 100k forward RGB `passed: false`)

The protocol (`benchmarks/configs/renderer_mi300x_v1.json`) says the FP64 CPU reference is the correctness oracle on
the small cases only. The benchmark already applied that to gradients (`agreement_only` on non-gate cases), but the
forward rows carried `passed` on every case. They are now classified the same way (`benchmarks/renderer/classify.py`):

| case | oracle role | acceptance result (both backends) |
|---|---|---|
| N=16, 32² (`gate: true`) | acceptance gate | 16/16 oracle metrics pass (3 forward + 5 gradient families × 2 backends) |
| N=1k, 10k, 100k | diagnostic | raw value, tolerance and `within_tolerance` kept; `passed: null` |

The 100k forward RGB error stays recorded as **1.0456162494575927e-3** against the unchanged **2e-4** tolerance
(`within_tolerance: false`, `recorded_passed: false` kept visible). The strict small-case gate is unchanged.

## 2. What the 100k RGB error is

Recorded facts:
- gsplat and berkelium_rocm report bitwise-identical forward errors against the oracle in every case (RGB, alpha,
  depth). By construction they share the forward: the Berkelium path calls gsplat's
  `rasterize_to_pixels_3dgs_fwd` (`render/berkelium_ext.py`) and only replaces the backward. Identical errors are
  consistent with identical forward outputs but do not prove them (see §4).
- At 100k: RGB 1.0456e-3, alpha 3.18e-5, depth 6.3e-4 (within its relative tolerance).

CPU diagnosis (Berkelium reference renderer, same scenes/cameras/rules; **not** a GPU measurement):

| 100k, 512² vs FP64 oracle | RGB max abs | alpha max abs | pixels with RGB err > 2e-4 |
|---|---|---|---|
| C: FP32 projection + FP32 compositing (independent FP32 implementation) | 1.0450e-3 | 3.1839362198748944e-5 | 2 / 262,144 |
| B: FP32 projection, FP64 compositing | 1.0451e-3 | 3.1853e-5 | 2 |
| D: FP64 values, FP32 pair list/order only | 1.9e-5 | 0 | 0 |

- The independent FP32 implementation reproduces the GPU numbers (alpha error identical to the last printed digit,
  RGB within 0.06%). Smaller cases also match the GPU scale (1k: 2.37e-6 vs 1.85e-6; 10k: 3.47e-6 vs 3.14e-6).
- The error comes from FP32 projection, not compositing arithmetic (B ≈ C) and not depth ordering (D is small).
- At the worst pixel (y=153, x=263), Gaussian #31767 has alpha 0.0039215033 in FP64 and 0.0039215983 from the FP32
  projection, on either side of the 1/255 = 0.0039215686 skip threshold, at transmittance 0.983. The FP32 path
  includes it, the FP64 oracle skips it. A single flip like this moves a pixel by up to ~(1/255)·T·|Δcolour|.

Conclusion: the 100k RGB value measures a near-threshold decision flip that any FP32 implementation of profile
`gsplat_v1` exhibits on this workload; it is not evidence of a gsplat or Berkelium defect, and it is not an
acceptance failure under the protocol. That the GPU flips the same Gaussian is plausible but not verified (no
per-pixel GPU trace was recorded → `not_run`).

## 3. Approval metadata

The result stores the experiment config verbatim, so `"approved": false` is the static config value. The runner at
that time did not record the CLI flag. The reclassified report shows `cli_i_approve_gpu_cost: not_recorded` and notes
that a completed run of an `approved: false` experiment implies the flag (an inference from runner code, not a
recorded fact; the run's tree was `0700bd9a…+dirty`). Future runs record an `authorization` block
(`static_config_approved`, `cli_i_approve_gpu_cost`, `authorized_by`) and never rewrite `approved`.

## 4. Direct gsplat-vs-Berkelium comparison

Added to the benchmark for future runs (`cross_backend` per case: forward RGB/alpha/depth and per-family gradient
max-abs difference, bitwise-equal flag, differing-entry count). For E-R1 it is **not_run**: forward outputs were not
recorded, and it is not inferred from the oracle errors. Validating it needs a GPU rerun (not done here).

## 5. Other recorded E-R1 results (unchanged, for context)

| N | backward ms gsplat → Berkelium | iteration ms | peak MiB | unique gradient hashes / 10 runs |
|---|---|---|---|---|
| 16 | 0.419 → 0.599 (1.43×) | 1.353 → 1.571 | 0.15 → 0.17 | gsplat 3, Berkelium 1 |
| 1k | 0.645 → 1.177 (1.82×) | 1.625 → 2.172 | 2.5 → 2.7 | 10, 1 |
| 10k | 0.904 → 1.849 (2.05×) | 2.015 → 2.962 | 13.7 → 14.6 | 10, 1 |
| 100k | 1.486 → 3.408 (2.29×) | 2.975 → 4.946 | 99.7 → 110.8 | 10, 1 |

Berkelium's backward is bitwise repeatable in every case; gsplat's is not. It costs 1.4–2.3× gsplat's backward time.
Fixed-budget fit PSNR is 22.83 dB (gsplat) vs 22.91 dB (Berkelium); each backend's ground truth is rendered by that
same backend, so this measures optimisation, not geometric accuracy.

## 6. MI300X rerun, 2026-10-15 (`E-R1_mi300x.json`, history `E-R1_mi300x.20261015T160753Z.json`)

Conditions: one AMD Instinct MI300X VF, PyTorch 2.8.0+rocm7.0.2 (HIP 7.0.51831), Python 3.12.3, AMD gsplat wheel
`amd_gsplat 1.5.3+30c1e78` (pip, `pypi.amd.com/rocm-7.0.0`), Berkelium kernel JIT-built with hipcc from
`/opt/rocm/core-10.1` (the doctor reports no `/opt/rocm/.info/version`; the hipcc that built the kernel is not the
ROCm 7.0 toolchain torch was built against — recorded, not hidden). Source tree `5a17478…+dirty` (dirty = result files
of the failed attempt below). Same protocol `renderer-mi300x-v1` (fp32, profile gsplat_v1, 10 warmup + 50 timed
iterations, 10 determinism repeats). Authorization recorded at run time: `cli_flag` (config `approved` unchanged).

A first attempt the same day (`E-R1_mi300x.20261015T150902Z.json`, `status: failed`) is kept: the Berkelium extension
JIT build left no `berkelium_det_bwd.so` (ImportError) on a fresh venv; per-case failures were recorded but the
unguarded fit-quality step then aborted the run and discarded the case data. The ROCm tests (`pytest -m hip`, 4 passed)
built the extension into the same cache, after which this run completed. The benchmark now records fit failures per
backend instead of aborting (`7935cd7`).

| N (pairs) | backward ms gsplat → Berkelium | p95 | iteration ms | peak MiB | unique gradient hashes / 10 |
|---|---|---|---|---|---|
| 16 (31) | 0.476 → 0.629 (1.32×) | 0.675 → 0.738 | 1.462 → 1.561 | 0.2 → 0.2 | 2 → **1** |
| 1k (2,900) | 0.576 → 1.017 (1.77×) | 0.599 → 1.101 | 1.580 → 2.032 | 2.5 → 2.7 | 10 → **1** |
| 10k (27,907) | 0.780 → 1.517 (1.95×) | 0.809 → 1.552 | 1.877 → 2.621 | 13.7 → 14.6 | 10 → **1** |
| 100k (265,653) | 1.258 → 2.781 (2.21×) | 1.283 → 2.824 | 2.641 → 4.177 | 99.7 → 110.8 | 10 → **1** |

- Acceptance gates (N=16, both backends): 16/16 pass; FP64 finite-difference gradcheck of the oracle: pass.
- **Direct comparison (was `not_run` in §4, now measured):** Berkelium's forward RGB/alpha/depth is **bitwise equal**
  to gsplat's in every case (up to 786,432 RGB entries at 100k) — it calls gsplat's forward, and this confirms it.
  Gradients differ from gsplat's by at most 2.1e-7 – 4.1e-7 relative to the largest gradient of each family: float32
  rounding from a different (fixed vs atomic) accumulation order, not a different computation.
- Determinism: Berkelium's gradients are bit-identical across 10 repeats in every case; gsplat's are not (2–10 hashes).
- Cost: Berkelium's backward is 1.32–2.21× gsplat's (10-13 run: 1.43–2.29×); the forward is unchanged.
- Fixed-budget fit PSNR (each backend renders its own GT): gsplat 22.91 dB, Berkelium 22.97 dB.
- The 100k forward RGB error vs the FP64 oracle remains diagnostic (§1–2).

Regression coverage for these conditions: `services/neural/tests/test_gpu_rocm.py` (`-m hip`: forward parity vs FP64,
gradient agreement, 10-repeat bitwise determinism on the GPU) — 4 passed on the MI300X before this run; the HIP-CPU
tests (`test_hip_kernel_cpu.py`) run the same kernel source on CPU in every CI-style run.
