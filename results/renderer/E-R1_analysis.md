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
