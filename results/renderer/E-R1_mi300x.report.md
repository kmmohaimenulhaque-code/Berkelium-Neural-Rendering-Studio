# Benchmark report: E-R1 — AMD gsplat vs Berkelium deterministic backward on one MI300X

- result: `results/renderer/E-R1_mi300x.json`  config hash `e542f7946faf6405`
- commit `5a17478a71ae035929ac40694ef54292f8551b3d+dirty` · torch 2.8.0+rocm7.0.2.git245bf6ed (hip=7.0.51831-7c9236b16, cuda=None) · GPU visible: True · Linux-6.8.0-146-generic-x86_64-with-glibc2.39
- method: `benchmarks/renderer/bench_renderer.py` · started 2026-10-15T16:07:53Z

Semantics: All backends use profile gsplat_v1: alpha=min(0.999, o*exp(-sigma)); skip if sigma<0 or alpha<1/255; stop exclusively when T*(1-alpha)<=1e-4; eps2d=0.3; FOV-clipped Jacobian (1.3x); opacity-aware 3.33-sigma tile radius; depth-sorted per tile. Known differences: equal-depth ties are broken by Gaussian index in the reference and by sort stability in gsplat; the reference differentiates the forward with autograd (cumprod), while gsplat and Berkelium recover transmittance in backward by division (T /= 1-alpha); gsplat accumulates cross-tile gradients with float atomicAdd, Berkelium with fixed-order per-intersection reductions.

Baseline backend **gsplat** vs candidate **berkelium_rocm**.

- global gate `oracle_fd_gradcheck_fp64`: pass (torch.autograd.gradcheck central differences eps=1e-6 atol=1e-6 rtol=1e-4, 5 families)

### Authorization

- static config `approved`: False
- per-run CLI `--i-approve-gpu-cost`: True
- authorized by: cli_flag
- note: per-run authorization asserted by the operator at invocation (--i-approve-gpu-cost); the experiment config was not modified

### FP64 oracle role per case (oracle-role-v1)

The FP64 CPU oracle is an acceptance gate only on cases marked `gate` (protocol). Elsewhere its errors are diagnostic: values and tolerances are kept, `passed` is null, and a value beyond the tolerance is not an acceptance failure (see `benchmarks/renderer/classify.py`).

| case | oracle role | forward oracle errors bitwise-identical across backends |
|---|---|---|
| N=16 32px | acceptance_gate | forward_rgb_max_abs_err: True, forward_alpha_max_abs_err: True, forward_depth_max_abs_err_alpha>0.5: True |
| N=1000 128px | diagnostic | forward_rgb_max_abs_err: True, forward_alpha_max_abs_err: True, forward_depth_max_abs_err_alpha>0.5: True |
| N=10000 256px | diagnostic | forward_rgb_max_abs_err: True, forward_alpha_max_abs_err: True, forward_depth_max_abs_err_alpha>0.5: True |
| N=100000 512px | diagnostic | forward_rgb_max_abs_err: True, forward_alpha_max_abs_err: True, forward_depth_max_abs_err_alpha>0.5: True |

- acceptance gates (both backends, all oracle metrics on gate cases): 16 pass, 0 fail
- diagnostic beyond tolerance (not a failure): N=1000 gsplat `grad_means_max_abs_err` = 0.0006746 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=1000 berkelium_rocm `grad_means_max_abs_err` = 0.0006746 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=10000 gsplat `grad_means_max_abs_err` = 0.01626 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=10000 gsplat `grad_log_scales_max_abs_err` = 0.0003022 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=10000 gsplat `grad_quats_max_abs_err` = 0.0002622 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=10000 gsplat `grad_opacity_logits_max_abs_err` = 0.0001118 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=10000 gsplat `grad_color_logits_max_abs_err` = 0.0002466 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=10000 berkelium_rocm `grad_means_max_abs_err` = 0.01629 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=10000 berkelium_rocm `grad_log_scales_max_abs_err` = 0.0003021 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=10000 berkelium_rocm `grad_quats_max_abs_err` = 0.0002619 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=10000 berkelium_rocm `grad_opacity_logits_max_abs_err` = 0.0001117 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=10000 berkelium_rocm `grad_color_logits_max_abs_err` = 0.0002467 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=100000 gsplat `forward_rgb_max_abs_err` = 0.001046 (tol 0.0002)
- diagnostic beyond tolerance (not a failure): N=100000 gsplat `grad_means_max_abs_err` = 0.2449 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=100000 gsplat `grad_log_scales_max_abs_err` = 0.004314 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=100000 gsplat `grad_quats_max_abs_err` = 0.004218 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=100000 gsplat `grad_opacity_logits_max_abs_err` = 0.0001479 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=100000 gsplat `grad_color_logits_max_abs_err` = 0.001035 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=100000 berkelium_rocm `forward_rgb_max_abs_err` = 0.001046 (tol 0.0002)
- diagnostic beyond tolerance (not a failure): N=100000 berkelium_rocm `grad_means_max_abs_err` = 0.2449 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=100000 berkelium_rocm `grad_log_scales_max_abs_err` = 0.004315 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=100000 berkelium_rocm `grad_quats_max_abs_err` = 0.004219 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=100000 berkelium_rocm `grad_opacity_logits_max_abs_err` = 0.0001479 (tol 2e-05+0.002|ref|)
- diagnostic beyond tolerance (not a failure): N=100000 berkelium_rocm `grad_color_logits_max_abs_err` = 0.001035 (tol 2e-05+0.002|ref|)
- identical oracle errors are consistent with identical forward outputs but do not prove them; the direct `forward_*_diff` rows do (or are `not_run`).

## correctness

| case | metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|---|
| N=16 32px | forward_alpha_max_abs_err | 5.39e-07 | 5.39e-07 | 0 | 0 | pass |
| N=16 32px | forward_depth_max_abs_err_alpha>0.5 | 6.282e-07 | 6.282e-07 | 0 | 0 | pass |
| N=16 32px | forward_rgb_max_abs_err | 4.483e-07 | 4.483e-07 | 0 | 0 | pass |
| N=16 32px | grad_color_logits_max_abs_err | 3.543e-07 | 3.841e-07 | 2.98e-08 | 0.08412 | pass |
| N=16 32px | grad_log_scales_max_abs_err | 1.685e-06 | 1.208e-06 | -4.768e-07 | -0.283 | pass |
| N=16 32px | grad_means_max_abs_err | 2.398e-05 | 2.35e-05 | -4.768e-07 | -0.01989 | pass |
| N=16 32px | grad_opacity_logits_max_abs_err | 6.203e-07 | 5.011e-07 | -1.192e-07 | -0.1922 | pass |
| N=16 32px | grad_quats_max_abs_err | 1.404e-06 | 1.046e-06 | -3.576e-07 | -0.2547 | pass |
| N=16 32px | gradient_bitwise_repeatable | 0 | 1 | 1 | null | pass |
| N=16 32px | forward_rgb_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=16 32px | forward_alpha_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=16 32px | forward_depth_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=16 32px | grad_means_max_abs_diff_vs_gsplat | null | 3.815e-06 | null | null | info: 40/48 entries differ (direct comparison, no oracle) |
| N=16 32px | grad_log_scales_max_abs_diff_vs_gsplat | null | 4.768e-07 | null | null | info: 35/48 entries differ (direct comparison, no oracle) |
| N=16 32px | grad_quats_max_abs_diff_vs_gsplat | null | 3.576e-07 | null | null | info: 47/64 entries differ (direct comparison, no oracle) |
| N=16 32px | grad_opacity_logits_max_abs_diff_vs_gsplat | null | 1.192e-07 | null | null | info: 11/16 entries differ (direct comparison, no oracle) |
| N=16 32px | grad_color_logits_max_abs_diff_vs_gsplat | null | 5.96e-08 | null | null | info: 39/48 entries differ (direct comparison, no oracle) |
| N=1000 128px | forward_alpha_max_abs_err | 1.852e-06 | 1.852e-06 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (within tol 0.0002) |
| N=1000 128px | forward_depth_max_abs_err_alpha>0.5 | 2.029e-06 | 2.029e-06 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (within tol 1e-4+2e-4|ref| (spec valid-depth)) |
| N=1000 128px | forward_rgb_max_abs_err | 1.854e-06 | 1.854e-06 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (within tol 0.0002) |
| N=1000 128px | grad_color_logits_max_abs_err | 4.793e-05 | 4.77e-05 | -2.384e-07 | -0.004974 | diagnostic: FP64 oracle not a gate for this case (within tol 2e-05+0.002|ref|; 0/3000 entries beyond spec tol) |
| N=1000 128px | grad_log_scales_max_abs_err | 3.469e-05 | 3.457e-05 | -1.192e-07 | -0.003437 | diagnostic: FP64 oracle not a gate for this case (within tol 2e-05+0.002|ref|; 0/3000 entries beyond spec tol) |
| N=1000 128px | grad_means_max_abs_err | 0.0006746 | 0.0006746 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 1/3000 entries beyond spec tol) |
| N=1000 128px | grad_opacity_logits_max_abs_err | 4.574e-06 | 4.485e-06 | -8.941e-08 | -0.01955 | diagnostic: FP64 oracle not a gate for this case (within tol 2e-05+0.002|ref|; 0/1000 entries beyond spec tol) |
| N=1000 128px | grad_quats_max_abs_err | 1.493e-05 | 1.487e-05 | -5.96e-08 | -0.003993 | diagnostic: FP64 oracle not a gate for this case (within tol 2e-05+0.002|ref|; 0/4000 entries beyond spec tol) |
| N=1000 128px | gradient_bitwise_repeatable | 0 | 1 | 1 | null | pass |
| N=1000 128px | forward_rgb_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=1000 128px | forward_alpha_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=1000 128px | forward_depth_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=1000 128px | grad_means_max_abs_diff_vs_gsplat | null | 2.289e-05 | null | null | info: 2475/3000 entries differ (direct comparison, no oracle) |
| N=1000 128px | grad_log_scales_max_abs_diff_vs_gsplat | null | 1.192e-06 | null | null | info: 2509/3000 entries differ (direct comparison, no oracle) |
| N=1000 128px | grad_quats_max_abs_diff_vs_gsplat | null | 1.848e-06 | null | null | info: 3400/4000 entries differ (direct comparison, no oracle) |
| N=1000 128px | grad_opacity_logits_max_abs_diff_vs_gsplat | null | 3.874e-07 | null | null | info: 782/1000 entries differ (direct comparison, no oracle) |
| N=1000 128px | grad_color_logits_max_abs_diff_vs_gsplat | null | 3.427e-07 | null | null | info: 2272/3000 entries differ (direct comparison, no oracle) |
| N=10000 256px | forward_alpha_max_abs_err | 4.93e-06 | 4.93e-06 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (within tol 0.0002) |
| N=10000 256px | forward_depth_max_abs_err_alpha>0.5 | 2.672e-06 | 2.672e-06 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (within tol 1e-4+2e-4|ref| (spec valid-depth)) |
| N=10000 256px | forward_rgb_max_abs_err | 3.135e-06 | 3.135e-06 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (within tol 0.0002) |
| N=10000 256px | grad_color_logits_max_abs_err | 0.0002466 | 0.0002467 | 8.941e-08 | 0.0003626 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 21/30000 entries beyond spec tol) |
| N=10000 256px | grad_log_scales_max_abs_err | 0.0003022 | 0.0003021 | -1.192e-07 | -0.0003944 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 15/30000 entries beyond spec tol) |
| N=10000 256px | grad_means_max_abs_err | 0.01626 | 0.01629 | 2.289e-05 | 0.001407 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 125/30000 entries beyond spec tol) |
| N=10000 256px | grad_opacity_logits_max_abs_err | 0.0001118 | 0.0001117 | -1.118e-07 | -0.001 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 3/10000 entries beyond spec tol) |
| N=10000 256px | grad_quats_max_abs_err | 0.0002622 | 0.0002619 | -3.129e-07 | -0.001193 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 7/40000 entries beyond spec tol) |
| N=10000 256px | gradient_bitwise_repeatable | 0 | 1 | 1 | null | pass |
| N=10000 256px | forward_rgb_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=10000 256px | forward_alpha_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=10000 256px | forward_depth_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=10000 256px | grad_means_max_abs_diff_vs_gsplat | null | 7.629e-05 | null | null | info: 25316/30000 entries differ (direct comparison, no oracle) |
| N=10000 256px | grad_log_scales_max_abs_diff_vs_gsplat | null | 1.431e-06 | null | null | info: 25225/30000 entries differ (direct comparison, no oracle) |
| N=10000 256px | grad_quats_max_abs_diff_vs_gsplat | null | 1.669e-06 | null | null | info: 34539/40000 entries differ (direct comparison, no oracle) |
| N=10000 256px | grad_opacity_logits_max_abs_diff_vs_gsplat | null | 3.576e-07 | null | null | info: 8059/10000 entries differ (direct comparison, no oracle) |
| N=10000 256px | grad_color_logits_max_abs_diff_vs_gsplat | null | 6.184e-07 | null | null | info: 22902/30000 entries differ (direct comparison, no oracle) |
| N=100000 512px | forward_alpha_max_abs_err | 3.184e-05 | 3.184e-05 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (within tol 0.0002) |
| N=100000 512px | forward_depth_max_abs_err_alpha>0.5 | 0.0006344 | 0.0006344 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (within tol 1e-4+2e-4|ref| (spec valid-depth)) |
| N=100000 512px | forward_rgb_max_abs_err | 0.001046 | 0.001046 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (beyond tol 0.0002) |
| N=100000 512px | grad_color_logits_max_abs_err | 0.001035 | 0.001035 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 316/300000 entries beyond spec tol) |
| N=100000 512px | grad_log_scales_max_abs_err | 0.004314 | 0.004315 | 1.49e-07 | 3.454e-05 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 291/300000 entries beyond spec tol) |
| N=100000 512px | grad_means_max_abs_err | 0.2449 | 0.2449 | 1.526e-05 | 6.23e-05 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 2868/300000 entries beyond spec tol) |
| N=100000 512px | grad_opacity_logits_max_abs_err | 0.0001479 | 0.0001479 | 0 | 0 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 36/100000 entries beyond spec tol) |
| N=100000 512px | grad_quats_max_abs_err | 0.004218 | 0.004219 | 2.384e-07 | 5.652e-05 | diagnostic: FP64 oracle not a gate for this case (beyond tol 2e-05+0.002|ref|; 214/400000 entries beyond spec tol) |
| N=100000 512px | gradient_bitwise_repeatable | 0 | 1 | 1 | null | pass |
| N=100000 512px | forward_rgb_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=100000 512px | forward_alpha_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=100000 512px | forward_depth_max_abs_diff_vs_gsplat | null | 0 | null | null | info: bitwise equal (direct comparison, no oracle) |
| N=100000 512px | grad_means_max_abs_diff_vs_gsplat | null | 0.0001221 | null | null | info: 203365/300000 entries differ (direct comparison, no oracle) |
| N=100000 512px | grad_log_scales_max_abs_diff_vs_gsplat | null | 2.384e-06 | null | null | info: 200831/300000 entries differ (direct comparison, no oracle) |
| N=100000 512px | grad_quats_max_abs_diff_vs_gsplat | null | 2.995e-06 | null | null | info: 274074/400000 entries differ (direct comparison, no oracle) |
| N=100000 512px | grad_opacity_logits_max_abs_diff_vs_gsplat | null | 3.278e-07 | null | null | info: 64090/100000 entries differ (direct comparison, no oracle) |
| N=100000 512px | grad_color_logits_max_abs_diff_vs_gsplat | null | 6.352e-07 | null | null | info: 179609/300000 entries differ (direct comparison, no oracle) |

## workload

| case | metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|---|
| N=16 32px | max_pairs_per_tile | 9 | 9 | 0 | 0 | info |
| N=16 32px | n_gaussians | 16 | 16 | 0 | 0 | info |
| N=16 32px | n_pairs | 31 | 31 | 0 | 0 | info |
| N=16 32px | n_visible | 16 | 16 | 0 | 0 | info |
| N=1000 128px | max_pairs_per_tile | 221 | 221 | 0 | 0 | info |
| N=1000 128px | n_gaussians | 1000 | 1000 | 0 | 0 | info |
| N=1000 128px | n_pairs | 2900 | 2900 | 0 | 0 | info |
| N=1000 128px | n_visible | 1000 | 1000 | 0 | 0 | info |
| N=10000 256px | max_pairs_per_tile | 488 | 488 | 0 | 0 | info |
| N=10000 256px | n_gaussians | 10000 | 10000 | 0 | 0 | info |
| N=10000 256px | n_pairs | 27907 | 27907 | 0 | 0 | info |
| N=10000 256px | n_visible | 10000 | 10000 | 0 | 0 | info |
| N=100000 512px | max_pairs_per_tile | 1236 | 1236 | 0 | 0 | info |
| N=100000 512px | n_gaussians | 100000 | 100000 | 0 | 0 | info |
| N=100000 512px | n_pairs | 265653 | 265653 | 0 | 0 | info |
| N=100000 512px | n_visible | 100000 | 100000 | 0 | 0 | info |

## performance

| case | metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|---|
| N=16 32px | backward_latency | 0.4764 | 0.6295 | 0.153 | 0.3212 | info |
| N=16 32px | forward_latency | 0.7267 | 0.6884 | -0.0383 | -0.0527 | info |
| N=16 32px | iteration_latency | 1.462 | 1.561 | 0.09888 | 0.06764 | info |
| N=16 32px | peak_memory | 0.1519 | 0.166 | 0.01416 | 0.09325 | info |
| N=1000 128px | backward_latency | 0.5757 | 1.017 | 0.441 | 0.7661 | info |
| N=1000 128px | forward_latency | 0.74 | 0.7467 | 0.006741 | 0.00911 | info |
| N=1000 128px | iteration_latency | 1.58 | 2.032 | 0.4522 | 0.2863 | info |
| N=1000 128px | peak_memory | 2.51 | 2.698 | 0.188 | 0.0749 | info |
| N=10000 256px | backward_latency | 0.7796 | 1.517 | 0.7373 | 0.9458 | info |
| N=10000 256px | forward_latency | 0.8227 | 0.8361 | 0.0134 | 0.01629 | info |
| N=10000 256px | iteration_latency | 1.877 | 2.621 | 0.7438 | 0.3963 | info |
| N=10000 256px | peak_memory | 13.67 | 14.59 | 0.9214 | 0.0674 | info |
| N=100000 512px | backward_latency | 1.258 | 2.781 | 1.523 | 1.211 | info |
| N=100000 512px | forward_latency | 1.103 | 1.112 | 0.009315 | 0.008447 | info |
| N=100000 512px | iteration_latency | 2.641 | 4.177 | 1.537 | 0.5819 | info |
| N=100000 512px | peak_memory | 99.65 | 110.8 | 11.19 | 0.1123 | info |

## quality

| case | metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|---|
| fit | heldout_view_psnr_after_fit | 22.91 | 22.97 | 0.05381 | 0.002348 | improved |
