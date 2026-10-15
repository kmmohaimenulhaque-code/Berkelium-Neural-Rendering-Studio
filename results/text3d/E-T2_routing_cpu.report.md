# Benchmark report: E-T2 — Text-to-3D adapter routing (adapter=auto) vs always-on adapter and base

- result: `../results/text3d/E-T2_routing_cpu.json`  config hash `1e9acca52fba69e4`
- commit `bf2dbfffae81764a69709ac46898c8e25960e189+dirty` · torch 2.14.1+cpu (hip=None, cuda=None) · GPU visible: False · Linux-6.8.0-1064-azure-x86_64-with-glibc2.39
- method: `benchmarks/text3d/bench_text3d.py` · started 2026-10-15T17:19:48Z


## routed_vs_always_adapter_on_base_prompts (split `test`): base+lora → routed

| metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|
| success_rate | 1 | 1 | 0 | 0 | equal |
| part_count_accuracy | 0.2812 | 0.4688 | 0.1875 | 0.6667 | improved |
| color_accuracy | 0.9375 | 0.9375 | 0 | 0 | equal |
| chamfer_l1 | 0.1316 | 0.08986 | -0.04178 | -0.3174 | improved |
| chamfer_l2 | 0.02976 | 0.01412 | -0.01564 | -0.5257 | improved |
| normal_consistency | 0.7037 | 0.7235 | 0.01978 | 0.02811 | improved |
| fscore@0.02 | 0.1019 | 0.1652 | 0.06322 | 0.6203 | improved |
| fscore@0.05 | 0.2517 | 0.3958 | 0.1442 | 0.5728 | improved |
| view_psnr | 12.07 | 13.69 | 1.621 | 0.1344 | improved |
| view_ssim | 0.5879 | 0.6355 | 0.04766 | 0.08106 | improved |
| color_mae | 0.04045 | 0.0422 | 0.001759 | 0.04348 | regressed |
| seconds | 2.943 | 2.192 | -0.7515 | -0.2553 | improved |
| lpips | null | null | null | null | not_comparable (requires pretrained LPIPS network weights (download not approved)) |
| peak_rss | 899.7 | 899.7 | 0 | 0 | info |
| paired_delta_chamfer_l1 | null | -0.04178 | -0.04178 | null | CI excludes 0 (95% CI [-0.05452, -0.02947], n=32) |
| paired_delta_fscore@0.05 | null | 0.1442 | 0.1442 | null | CI excludes 0 (95% CI [0.08456, 0.2075], n=32) |
| paired_delta_view_psnr | null | 1.621 | 1.621 | null | CI excludes 0 (95% CI [1.057, 2.192], n=32) |
| paired_delta_normal_consistency | null | 0.01978 | 0.01978 | null | CI includes 0 (95% CI [-0.0049, 0.04438], n=32) |

## routed_vs_base_on_base_prompts (split `test`): base → routed

| metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|
| success_rate | 1 | 1 | 0 | 0 | equal |
| part_count_accuracy | 0.4688 | 0.4688 | 0 | 0 | equal |
| color_accuracy | 0.9375 | 0.9375 | 0 | 0 | equal |
| chamfer_l1 | 0.08986 | 0.08986 | 0 | 0 | equal |
| chamfer_l2 | 0.01412 | 0.01412 | 0 | 0 | equal |
| normal_consistency | 0.7235 | 0.7235 | 0 | 0 | equal |
| fscore@0.02 | 0.1652 | 0.1652 | 0 | 0 | equal |
| fscore@0.05 | 0.3958 | 0.3958 | 0 | 0 | equal |
| view_psnr | 13.69 | 13.69 | 0 | 0 | equal |
| view_ssim | 0.6355 | 0.6355 | 0 | 0 | equal |
| color_mae | 0.0422 | 0.0422 | 0 | 0 | equal |
| seconds | 1.93 | 1.789 | -0.1407 | -0.07291 | improved |
| lpips | null | null | null | null | not_comparable (requires pretrained LPIPS network weights (download not approved)) |
| peak_rss | 899.7 | 899.7 | 0 | 0 | info |
| paired_delta_chamfer_l1 | null | 0 | 0 | null | CI includes 0 (95% CI [0, 0], n=32) |
| paired_delta_fscore@0.05 | null | 0 | 0 | null | CI includes 0 (95% CI [0, 0], n=32) |
| paired_delta_view_psnr | null | 0 | 0 | null | CI includes 0 (95% CI [0, 0], n=32) |
| paired_delta_normal_consistency | null | 0 | 0 | null | CI includes 0 (95% CI [0, 0], n=32) |

## routed_vs_base_on_capsule_prompts (split `adapter_test`): base → routed

| metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|
| success_rate | 1 | 1 | 0 | 0 | equal |
| part_count_accuracy | 0.5333 | 0.2667 | -0.2667 | -0.5 | regressed |
| color_accuracy | 0.95 | 0.9333 | -0.01667 | -0.01754 | regressed |
| chamfer_l1 | 0.1342 | 0.08111 | -0.05312 | -0.3957 | improved |
| chamfer_l2 | 0.02927 | 0.0125 | -0.01677 | -0.5729 | improved |
| normal_consistency | 0.6673 | 0.7854 | 0.118 | 0.1769 | improved |
| fscore@0.02 | 0.09431 | 0.1681 | 0.07383 | 0.7829 | improved |
| fscore@0.05 | 0.2484 | 0.4279 | 0.1794 | 0.7223 | improved |
| view_psnr | 11.83 | 14.07 | 2.236 | 0.189 | improved |
| view_ssim | 0.606 | 0.7001 | 0.09415 | 0.1554 | improved |
| color_mae | 0.04943 | 0.04804 | -0.001389 | -0.02811 | improved |
| seconds | 1.753 | 2.832 | 1.079 | 0.6154 | regressed |
| lpips | null | null | null | null | not_comparable (requires pretrained LPIPS network weights (download not approved)) |
| peak_rss | 899.7 | 1032 | 132.5 | 0.1473 | info |
| paired_delta_chamfer_l1 | null | -0.05312 | -0.05312 | null | CI excludes 0 (95% CI [-0.07114, -0.03571], n=30) |
| paired_delta_fscore@0.05 | null | 0.1794 | 0.1794 | null | CI excludes 0 (95% CI [0.1058, 0.2599], n=30) |
| paired_delta_view_psnr | null | 2.236 | 2.236 | null | CI excludes 0 (95% CI [1.462, 3.02], n=30) |
| paired_delta_normal_consistency | null | 0.118 | 0.118 | null | CI excludes 0 (95% CI [0.09233, 0.1458], n=30) |

## routed_vs_always_adapter_on_capsule_prompts (split `adapter_test`): base+lora → routed

| metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|
| success_rate | 1 | 1 | 0 | 0 | equal |
| part_count_accuracy | 0.2667 | 0.2667 | 0 | 0 | equal |
| color_accuracy | 0.9333 | 0.9333 | 0 | 0 | equal |
| chamfer_l1 | 0.08111 | 0.08111 | 0 | 0 | equal |
| chamfer_l2 | 0.0125 | 0.0125 | 0 | 0 | equal |
| normal_consistency | 0.7854 | 0.7854 | 0 | 0 | equal |
| fscore@0.02 | 0.1681 | 0.1681 | 0 | 0 | equal |
| fscore@0.05 | 0.4279 | 0.4279 | 0 | 0 | equal |
| view_psnr | 14.07 | 14.07 | 0 | 0 | equal |
| view_ssim | 0.7001 | 0.7001 | 0 | 0 | equal |
| color_mae | 0.04804 | 0.04804 | 0 | 0 | equal |
| seconds | 2.854 | 2.847 | -0.006767 | -0.002371 | improved |
| lpips | null | null | null | null | not_comparable (requires pretrained LPIPS network weights (download not approved)) |
| peak_rss | 1148 | 1267 | 119.2 | 0.1038 | info |
| paired_delta_chamfer_l1 | null | 0 | 0 | null | CI includes 0 (95% CI [0, 0], n=30) |
| paired_delta_fscore@0.05 | null | 0 | 0 | null | CI includes 0 (95% CI [0, 0], n=30) |
| paired_delta_view_psnr | null | 0 | 0 | null | CI includes 0 (95% CI [0, 0], n=30) |
| paired_delta_normal_consistency | null | 0 | 0 | null | CI includes 0 (95% CI [0, 0], n=30) |
