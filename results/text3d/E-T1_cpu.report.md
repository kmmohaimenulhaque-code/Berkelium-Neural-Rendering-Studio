# Benchmark report: E-T1 — Text-to-3D: base vs retrieval baseline; LoRA adapter vs pretrained base

- result: `results/text3d/E-T1_cpu.json`  config hash `06208be2d66f2c97`
- commit `d525d20dc33935f60663769d9599b9ed6c4d975d+dirty` · torch 2.14.1+cu130 (hip=None, cuda=13.0) · GPU visible: False · Linux-6.18.44-fc-v114-x86_64-with-glibc2.39
- method: `benchmarks/text3d/bench_text3d.py` · started 2026-10-13T15:59:19Z


## base_vs_retrieval (split `test`): retrieval[train] → base

| metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|
| success_rate | 1 | 1 | 0 | 0 | equal |
| part_count_accuracy | 0.6875 | 0.4688 | -0.2188 | -0.3182 | regressed |
| color_accuracy | 0.4688 | 0.9375 | 0.4688 | 1 | improved |
| chamfer_l1 | 0.1244 | 0.08986 | -0.03459 | -0.2779 | improved |
| chamfer_l2 | 0.02609 | 0.01412 | -0.01198 | -0.4591 | improved |
| normal_consistency | 0.7371 | 0.7235 | -0.01367 | -0.01854 | regressed |
| fscore@0.02 | 0.1239 | 0.1652 | 0.04126 | 0.333 | improved |
| fscore@0.05 | 0.3023 | 0.3958 | 0.09351 | 0.3093 | improved |
| view_psnr | 11.88 | 13.69 | 1.807 | 0.1521 | improved |
| view_ssim | 0.5895 | 0.6355 | 0.04602 | 0.07807 | improved |
| color_mae | 0.1969 | 0.0422 | -0.1547 | -0.7857 | improved |
| seconds | 0.07634 | 0.7916 | 0.7153 | 9.369 | regressed |
| lpips | null | null | null | null | not_comparable (requires pretrained LPIPS network weights (download not approved)) |
| peak_rss | 651.1 | 1002 | 350.7 | 0.5386 | info |
| paired_delta_chamfer_l1 | null | -0.03459 | -0.03459 | null | CI excludes 0 (95% CI [-0.05164, -0.01755], n=32) |
| paired_delta_fscore@0.05 | null | 0.09351 | 0.09351 | null | CI excludes 0 (95% CI [0.01197, 0.1732], n=32) |
| paired_delta_view_psnr | null | 1.807 | 1.807 | null | CI excludes 0 (95% CI [0.9888, 2.622], n=32) |
| paired_delta_normal_consistency | null | -0.01367 | -0.01367 | null | CI includes 0 (95% CI [-0.06204, 0.03376], n=32) |

## lora_vs_base (split `adapter_test`): base → base+lora

| metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|
| success_rate | 1 | 1 | 0 | 0 | equal |
| part_count_accuracy | 0.5333 | 0.2667 | -0.2667 | -0.5 | regressed |
| color_accuracy | 0.95 | 0.9333 | -0.01667 | -0.01754 | regressed |
| chamfer_l1 | 0.1342 | 0.08111 | -0.05312 | -0.3957 | improved |
| chamfer_l2 | 0.02927 | 0.0125 | -0.01677 | -0.5729 | improved |
| normal_consistency | 0.6673 | 0.7854 | 0.1181 | 0.1769 | improved |
| fscore@0.02 | 0.09431 | 0.1681 | 0.07382 | 0.7827 | improved |
| fscore@0.05 | 0.2484 | 0.4279 | 0.1794 | 0.7223 | improved |
| view_psnr | 11.83 | 14.07 | 2.236 | 0.189 | improved |
| view_ssim | 0.606 | 0.7001 | 0.09415 | 0.1554 | improved |
| color_mae | 0.04943 | 0.04804 | -0.001389 | -0.02811 | improved |
| seconds | 0.7505 | 1.239 | 0.4882 | 0.6505 | regressed |
| lpips | null | null | null | null | not_comparable (requires pretrained LPIPS network weights (download not approved)) |
| peak_rss | 1015 | 1144 | 128.7 | 0.1268 | info |
| paired_delta_chamfer_l1 | null | -0.05312 | -0.05312 | null | CI excludes 0 (95% CI [-0.07115, -0.03571], n=30) |
| paired_delta_fscore@0.05 | null | 0.1794 | 0.1794 | null | CI excludes 0 (95% CI [0.1058, 0.2599], n=30) |
| paired_delta_view_psnr | null | 2.236 | 2.236 | null | CI excludes 0 (95% CI [1.462, 3.02], n=30) |
| paired_delta_normal_consistency | null | 0.1181 | 0.1181 | null | CI excludes 0 (95% CI [0.09235, 0.1458], n=30) |

## lora_vs_retrieval_with_adapter_data (split `adapter_test`): retrieval[train+adapter_train] → base+lora

| metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|
| success_rate | 1 | 1 | 0 | 0 | equal |
| part_count_accuracy | 0.7667 | 0.2667 | -0.5 | -0.6522 | regressed |
| color_accuracy | 0.5667 | 0.9333 | 0.3667 | 0.6471 | improved |
| chamfer_l1 | 0.1112 | 0.08111 | -0.03006 | -0.2704 | improved |
| chamfer_l2 | 0.02463 | 0.0125 | -0.01213 | -0.4926 | improved |
| normal_consistency | 0.7486 | 0.7854 | 0.03679 | 0.04915 | improved |
| fscore@0.02 | 0.1427 | 0.1681 | 0.02544 | 0.1783 | improved |
| fscore@0.05 | 0.339 | 0.4279 | 0.08882 | 0.262 | improved |
| view_psnr | 12.57 | 14.07 | 1.502 | 0.1195 | improved |
| view_ssim | 0.66 | 0.7001 | 0.04009 | 0.06074 | improved |
| color_mae | 0.1884 | 0.04804 | -0.1404 | -0.7451 | improved |
| seconds | 0.04343 | 1.246 | 1.203 | 27.69 | regressed |
| lpips | null | null | null | null | not_comparable (requires pretrained LPIPS network weights (download not approved)) |
| peak_rss | 1144 | 1144 | 0 | 0 | info |
| paired_delta_chamfer_l1 | null | -0.03006 | -0.03006 | null | CI excludes 0 (95% CI [-0.04564, -0.01543], n=30) |
| paired_delta_fscore@0.05 | null | 0.08882 | 0.08882 | null | CI excludes 0 (95% CI [0.02361, 0.1481], n=30) |
| paired_delta_view_psnr | null | 1.502 | 1.502 | null | CI excludes 0 (95% CI [0.7448, 2.248], n=30) |
| paired_delta_normal_consistency | null | 0.03679 | 0.03679 | null | CI includes 0 (95% CI [-0.0107, 0.08545], n=30) |

## base_retention_after_adapter (split `test`): base → base+lora

| metric | baseline | candidate | abs diff | rel change | status |
|---|---|---|---|---|---|
| success_rate | 1 | 1 | 0 | 0 | equal |
| part_count_accuracy | 0.4688 | 0.2812 | -0.1875 | -0.4 | regressed |
| color_accuracy | 0.9375 | 0.9375 | 0 | 0 | equal |
| chamfer_l1 | 0.08986 | 0.1316 | 0.04178 | 0.4649 | regressed |
| chamfer_l2 | 0.01412 | 0.02976 | 0.01564 | 1.108 | regressed |
| normal_consistency | 0.7235 | 0.7037 | -0.01977 | -0.02733 | regressed |
| fscore@0.02 | 0.1652 | 0.1019 | -0.06322 | -0.3828 | regressed |
| fscore@0.05 | 0.3958 | 0.2517 | -0.1442 | -0.3642 | regressed |
| view_psnr | 13.69 | 12.07 | -1.621 | -0.1184 | regressed |
| view_ssim | 0.6355 | 0.5879 | -0.04766 | -0.07499 | regressed |
| color_mae | 0.0422 | 0.04045 | -0.001759 | -0.04167 | improved |
| seconds | 0.7497 | 1.255 | 0.5054 | 0.6742 | regressed |
| lpips | null | null | null | null | not_comparable (requires pretrained LPIPS network weights (download not approved)) |
| peak_rss | 1144 | 1150 | 6.336 | 0.005538 | info |
| paired_delta_chamfer_l1 | null | 0.04178 | 0.04178 | null | CI excludes 0 (95% CI [0.02947, 0.05452], n=32) |
| paired_delta_fscore@0.05 | null | -0.1442 | -0.1442 | null | CI excludes 0 (95% CI [-0.2075, -0.08456], n=32) |
| paired_delta_view_psnr | null | -1.621 | -1.621 | null | CI excludes 0 (95% CI [-2.192, -1.057], n=32) |
| paired_delta_normal_consistency | null | -0.01977 | -0.01977 | null | CI includes 0 (95% CI [-0.04438, 0.004925], n=32) |
