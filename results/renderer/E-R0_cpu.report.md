# Benchmark report: E-R0 — Renderer correctness on CPU (reference + Berkelium HIP kernel under HIP-CPU)

- result: `results/renderer/E-R0_cpu.json`  config hash `b1031174ff3f9eb3`
- commit `d525d20dc33935f60663769d9599b9ed6c4d975d+dirty` · torch 2.14.1+cu130 (hip=None, cuda=13.0) · GPU visible: False · Linux-6.18.44-fc-v114-x86_64-with-glibc2.39
- method: `benchmarks/renderer/bench_renderer.py` · started 2026-10-13T16:17:29Z

Semantics: All backends use profile gsplat_v1: alpha=min(0.999, o*exp(-sigma)); skip if sigma<0 or alpha<1/255; stop exclusively when T*(1-alpha)<=1e-4; eps2d=0.3; FOV-clipped Jacobian (1.3x); opacity-aware 3.33-sigma tile radius; depth-sorted per tile. Known differences: equal-depth ties are broken by Gaussian index in the reference and by sort stability in gsplat; the reference differentiates the forward with autograd (cumprod), while gsplat and Berkelium recover transmittance in backward by division (T /= 1-alpha); gsplat accumulates cross-tile gradients with float atomicAdd, Berkelium with fixed-order per-intersection reductions.

Baseline backend **reference_cpu** vs candidate **berkelium_hipcpu**.

- global gate `oracle_fd_gradcheck_fp64`: pass (torch.autograd.gradcheck central differences eps=1e-6 atol=1e-6 rtol=1e-4, 5 families)

## correctness

| case | metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|---|
| N=16 32px | forward_alpha_max_abs_err | 5.986e-07 | 5.986e-07 | 0 | 0 | pass |
| N=16 32px | forward_depth_max_abs_err_alpha>0.5 | 5.669e-07 | 5.669e-07 | 0 | 0 | pass |
| N=16 32px | forward_rgb_max_abs_err | 3.887e-07 | 3.887e-07 | 0 | 0 | pass |
| N=16 32px | grad_color_logits_max_abs_err | 3.841e-07 | 3.841e-07 | 0 | 0 | pass |
| N=16 32px | grad_log_scales_max_abs_err | 1.208e-06 | 1.208e-06 | 0 | 0 | pass |
| N=16 32px | grad_means_max_abs_err | 2.493e-05 | 2.493e-05 | 0 | 0 | pass |
| N=16 32px | grad_opacity_logits_max_abs_err | 5.011e-07 | 6.203e-07 | 1.192e-07 | 0.2379 | pass |
| N=16 32px | grad_quats_max_abs_err | 9.837e-07 | 8.529e-07 | -1.308e-07 | -0.1329 | pass |
| N=16 32px | gradient_bitwise_repeatable | 1 | 1 | 0 | null | pass |
| N=200 64px | forward_alpha_max_abs_err | 8.748e-07 | 8.748e-07 | 0 | 0 | pass |
| N=200 64px | forward_depth_max_abs_err_alpha>0.5 | 1.196e-06 | 1.196e-06 | 0 | 0 | pass |
| N=200 64px | forward_rgb_max_abs_err | 7.531e-07 | 7.531e-07 | 0 | 0 | pass |
| N=200 64px | grad_color_logits_max_abs_err | 1.274e-06 | 2.034e-05 | 1.906e-05 | 14.96 | info: not a gate (agreement_only; 0/600 entries beyond spec tol) |
| N=200 64px | grad_log_scales_max_abs_err | 4.26e-06 | 1.461e-05 | 1.035e-05 | 2.429 | info: not a gate (agreement_only; 0/600 entries beyond spec tol) |
| N=200 64px | grad_means_max_abs_err | 7.034e-05 | 0.0002488 | 0.0001785 | 2.538 | info: not a gate (agreement_only; 0/600 entries beyond spec tol) |
| N=200 64px | grad_opacity_logits_max_abs_err | 1.539e-06 | 1.495e-06 | -4.47e-08 | -0.02904 | info: not a gate (agreement_only; 0/200 entries beyond spec tol) |
| N=200 64px | grad_quats_max_abs_err | 1.142e-05 | 1.165e-05 | 2.384e-07 | 0.02089 | info: not a gate (agreement_only; 0/800 entries beyond spec tol) |
| N=200 64px | gradient_bitwise_repeatable | 1 | 1 | 0 | null | pass |
| N=1000 96px | forward_alpha_max_abs_err | 1.751e-06 | 1.751e-06 | 0 | 0 | pass |
| N=1000 96px | forward_depth_max_abs_err_alpha>0.5 | 1.777e-06 | 1.777e-06 | 0 | 0 | pass |
| N=1000 96px | forward_rgb_max_abs_err | 1.608e-06 | 1.608e-06 | 0 | 0 | pass |
| N=1000 96px | grad_color_logits_max_abs_err | 3.25e-06 | 5.117e-05 | 4.792e-05 | 14.75 | info: not a gate (agreement_only; 0/3000 entries beyond spec tol) |
| N=1000 96px | grad_log_scales_max_abs_err | 1.256e-05 | 4.854e-05 | 3.598e-05 | 2.864 | info: not a gate (agreement_only; 0/3000 entries beyond spec tol) |
| N=1000 96px | grad_means_max_abs_err | 0.0004871 | 0.0007538 | 0.0002667 | 0.5475 | info: not a gate (agreement_only; 0/3000 entries beyond spec tol) |
| N=1000 96px | grad_opacity_logits_max_abs_err | 3.117e-06 | 1.067e-05 | 7.555e-06 | 2.424 | info: not a gate (agreement_only; 0/1000 entries beyond spec tol) |
| N=1000 96px | grad_quats_max_abs_err | 2.766e-05 | 2.754e-05 | -1.192e-07 | -0.00431 | info: not a gate (agreement_only; 0/4000 entries beyond spec tol) |
| N=1000 96px | gradient_bitwise_repeatable | 1 | 1 | 0 | null | pass |
| N=3000 128px | forward_alpha_max_abs_err | 3.83e-06 | 3.83e-06 | 0 | 0 | pass |
| N=3000 128px | forward_depth_max_abs_err_alpha>0.5 | 2.64e-06 | 2.64e-06 | 0 | 0 | pass |
| N=3000 128px | forward_rgb_max_abs_err | 3.366e-06 | 3.366e-06 | 0 | 0 | pass |
| N=3000 128px | grad_color_logits_max_abs_err | 3.277e-06 | 0.0001337 | 0.0001304 | 39.81 | info: not a gate (agreement_only; 0/9000 entries beyond spec tol) |
| N=3000 128px | grad_log_scales_max_abs_err | 1.158e-05 | 8.96e-05 | 7.802e-05 | 6.738 | info: not a gate (agreement_only; 0/9000 entries beyond spec tol) |
| N=3000 128px | grad_means_max_abs_err | 0.0003903 | 0.009769 | 0.009379 | 24.03 | info: not a gate (agreement_only; 19/9000 entries beyond spec tol) |
| N=3000 128px | grad_opacity_logits_max_abs_err | 2.254e-06 | 5.787e-05 | 5.561e-05 | 24.67 | info: not a gate (agreement_only; 0/3000 entries beyond spec tol) |
| N=3000 128px | grad_quats_max_abs_err | 1.991e-05 | 7.052e-05 | 5.06e-05 | 2.541 | info: not a gate (agreement_only; 0/12000 entries beyond spec tol) |
| N=3000 128px | gradient_bitwise_repeatable | 1 | 1 | 0 | null | pass |

## workload

| case | metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|---|
| N=16 32px | max_pairs_per_tile | 9 | 9 | 0 | 0 | info |
| N=16 32px | n_gaussians | 16 | 16 | 0 | 0 | info |
| N=16 32px | n_pairs | 31 | 31 | 0 | 0 | info |
| N=16 32px | n_visible | 16 | 16 | 0 | 0 | info |
| N=200 64px | max_pairs_per_tile | 95 | 95 | 0 | 0 | info |
| N=200 64px | n_gaussians | 200 | 200 | 0 | 0 | info |
| N=200 64px | n_pairs | 509 | 509 | 0 | 0 | info |
| N=200 64px | n_visible | 200 | 200 | 0 | 0 | info |
| N=1000 96px | max_pairs_per_tile | 299 | 299 | 0 | 0 | info |
| N=1000 96px | n_gaussians | 1000 | 1000 | 0 | 0 | info |
| N=1000 96px | n_pairs | 2515 | 2515 | 0 | 0 | info |
| N=1000 96px | n_visible | 1000 | 1000 | 0 | 0 | info |
| N=3000 128px | max_pairs_per_tile | 470 | 470 | 0 | 0 | info |
| N=3000 128px | n_gaussians | 3000 | 3000 | 0 | 0 | info |
| N=3000 128px | n_pairs | 6984 | 6984 | 0 | 0 | info |
| N=3000 128px | n_visible | 3000 | 3000 | 0 | 0 | info |

## performance

| case | metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|---|
| N=16 32px | backward_latency | 2.505 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=16 32px | forward_latency | 3.57 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=16 32px | iteration_latency | 6.392 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=16 32px | peak_memory | 657 | 662.3 | 5.289 | 0.008051 | info |
| N=200 64px | backward_latency | 11.67 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=200 64px | forward_latency | 48.37 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=200 64px | iteration_latency | 62.97 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=200 64px | peak_memory | 676 | 676 | 0 | 0 | info |
| N=1000 96px | backward_latency | 132.5 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=1000 96px | forward_latency | 116.9 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=1000 96px | iteration_latency | 255.8 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=1000 96px | peak_memory | 739 | 739 | 0 | 0 | info |
| N=3000 128px | backward_latency | 159.8 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=3000 128px | forward_latency | 409.5 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=3000 128px | iteration_latency | 586.8 | null | null | null | not_comparable (HIP-CPU emulation: timing is not GPU performance) |
| N=3000 128px | peak_memory | 843.3 | 843.3 | 0 | 0 | info |

## quality

| case | metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|---|
| fit | heldout_view_psnr_after_fit | 21.65 | 21.64 | -0.01428 | -0.0006594 | regressed |
