# E-T4 — capsule adapter v2 (experience replay + separation margin)

Protocol berkelium-text3d-eval/1 at resolution 48; weights sha256 base `4989d90b206a`, v1 `fa5b2eae453c`, candidate `6a286f27dee8`. Rule written in `configs/adapters/textshape_lora_capsule_v2_replay.json` before training.

| model | split | n | part-count acc | CD-L1 | F@0.05 | view PSNR | colour acc |
|---|---|---|---|---|---|---|---|
| base | adapter_test | 30 | 0.53 | 0.134 | 0.248 | 11.83 | 0.95 |
| capsule-v1 | adapter_test | 30 | 0.27 | 0.081 | 0.428 | 14.07 | 0.93 |
| capsule-v1 | test | 32 | 0.28 | 0.132 | 0.252 | 12.07 | 0.94 |
| candidate | adapter_test | 30 | 0.40 | 0.099 | 0.387 | 13.76 | 0.90 |
| candidate | test | 32 | 0.34 | 0.133 | 0.338 | 13.03 | 0.88 |

| paired comparison (b − a) | metric | n | mean | 95% CI |
|---|---|---|---|---|
| candidate_vs_v1_adapter_test | chamfer_l1 | 30 | 0.0178 | [+0.0067, +0.0306] |
| candidate_vs_v1_adapter_test | fscore@0.05 | 30 | -0.0414 | [-0.1032, +0.0179] |
| candidate_vs_v1_adapter_test | view_psnr | 30 | -0.3158 | [-0.7672, +0.1320] |
| candidate_vs_base_adapter_test | chamfer_l1 | 30 | -0.0353 | [-0.0542, -0.0176] |
| candidate_vs_base_adapter_test | fscore@0.05 | 30 | 0.1381 | [+0.0741, +0.2086] |
| candidate_vs_base_adapter_test | view_psnr | 30 | 1.9206 | [+1.1734, +2.6957] |
| candidate_vs_v1_test_forced_on | chamfer_l1 | 32 | 0.0015 | [-0.0248, +0.0303] |
| candidate_vs_v1_test_forced_on | fscore@0.05 | 32 | 0.0865 | [+0.0214, +0.1529] |
| candidate_vs_v1_test_forced_on | view_psnr | 32 | 0.9655 | [+0.3314, +1.5809] |

Pre-registered checks: part_count_accuracy >= 0.53: FAIL; chamfer vs v1: CI upper <= +0.010: FAIL; chamfer vs base: CI upper < 0: pass

**Decision (by the rule): keep capsule-v1.**

