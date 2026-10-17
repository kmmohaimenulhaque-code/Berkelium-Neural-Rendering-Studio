# E-T3 open-world text-to-3D — full run

Config `benchmarks/configs/text3d_openworld_cpu_v1.json`; frozen prompts `benchmarks/configs/text3d_openworld_cpu_v1.prompts.json` (sha256 `42d1c2e8d5c445b6…`).
Scored on Linux-6.18.44-fc-v139-x86_64-with-glibc2.39; git bbcbefbd624a1a970891dddba865d2b433b450bd. Generators: berkelium-textshape, shap-e-text300m, zimage-triposr, zimage-triposr-v2. Training: none for native/Shap-E/zimage-triposr; see the anchor-LoRA model card for zimage-triposr-anchorlora.
Shap-E: text300M + vector decoder, 16 Karras steps, guidance 15.0, seed 0. Native: BerkeliumTextShape, resolution 48, adapter auto. zimage-triposr: Z-Image-Turbo (9 steps, CFG 0, 1024², canonical anchor template) -> colour key -> TripoSR (mc 256).

**Units: 296 done; frozen design 124 units (0 omitted).** Devices: cpu, cuda. Shap-E CPU and GPU units use the same weights, 16 steps, fp32 sampling and the same initial noise (seeded CPU generator); they are not bitwise identical (different kernels). zimage-triposr runs only on a GPU. Every row records its device.

zimage-triposr: 0 planned units not produced.

zimage-triposr-v2: 0 planned units not produced.

**Excluded run (infrastructure, re-run):** `shap-e-text300m|an elephant|0` failed(rc=-15) after 379.0 s on cpu: infrastructure: SIGTERM from the host memory watchdog while the operator ran test and check processes concurrently with this worker (system-wide MemAvailable fell under ~0.8 GB); not a model outcome. The unit is re-run. Log: `results/text3d/E-T3/units/shap-e-text300m/an_elephant__8b94b960__s0_killed_by_watchdog.log`.

## Cost by generator and device

| generator | device | hardware | units | valid | s/unit | wall h total | peak RSS MiB | peak GPU MiB |
|---|---|---|---|---|---|---|---|---|
| shap-e-text300m | cpu | cpu | 14 | 14 | 1061.6 | 4.32 | 2375 | — |
| shap-e-text300m | cuda | Tesla T4 | 66 | 66 | 31.6 | 1.87 | 2135 | 2047.4 |
| zimage-triposr | cuda | Tesla T4 | 80 | 37 | 147.8 | 3.28 | — | 13803.2 |
| zimage-triposr-v2 | cuda | Tesla T4 | 80 | 80 | 176.0 | 3.91 | — | 13803.2 |

Unit status counts: {'anchor_rejected': 43, 'ok': 253}.

## B1 — open-world prompts with real-object ground truth (Google Scanned Objects)

Alignment: unit extent + best of 24 axis rotations + ICP (same for every generator; generous: forgives pose, frame and scale). Chamfer-L1 and F-score in unit-extent units; lower CD / higher F is better.

| object | prompt | route | berkelium-textshape CD / F@0.05 | shap-e-text300m CD / F@0.05 | zimage-triposr CD / F@0.05 | zimage-triposr-v2 CD / F@0.05 |
|---|---|---|---|---|---|---|
| mug | a blue coffee mug | shap-e-text300m | 0.055 / 0.570 (measured) | 0.044 / 0.639 (measured) | 0.062 / 0.529 (measured) | 0.056 / 0.602 (measured) |
| teapot | a white porcelain teapot | shap-e-text300m | 0.092 / 0.272 (measured) | 0.122 / 0.131 (measured) | — / 0.000 (anchor_rejected) | 0.116 / 0.191 (measured) |
| pitcher | a white porcelain pitcher | shap-e-text300m | 0.041 / 0.714 (measured) | 0.028 / 0.843 (measured) | — / 0.000 (anchor_rejected) | 0.045 / 0.630 (measured) |
| bowl | a deep bowl | shap-e-text300m | 0.094 / 0.368 (measured) | 0.015 / 0.989 (measured) | — / 0.000 (anchor_rejected) | 0.034 / 0.810 (measured) |
| basket | a teal plastic easter basket | shap-e-text300m | 0.142 / 0.222 (measured) | 0.040 / 0.705 (measured) | — / 0.000 (anchor_rejected) | 0.063 / 0.554 (measured) |
| boot | a men's boot | shap-e-text300m | 0.039 / 0.679 (measured) | 0.034 / 0.799 (measured) | — / 0.000 (anchor_rejected) | 0.015 / 0.981 (measured) |
| boat shoe | a boat shoe | shap-e-text300m | 0.032 / 0.792 (measured) | 0.020 / 0.915 (measured) | 0.020 / 0.937 (measured) | 0.017 / 0.948 (measured) |
| school bus | a school bus | shap-e-text300m | 0.038 / 0.686 (measured) | 0.045 / 0.659 (measured) | 0.027 / 0.875 (measured) | 0.030 / 0.781 (measured) |
| airplane | a turboprop airplane with a pilot | shap-e-text300m | 0.070 / 0.460 (measured) | 0.072 / 0.371 (measured) | 0.068 / 0.413 (measured) | 0.079 / 0.360 (measured) |
| teddy bear | a beige teddy bear | shap-e-text300m | 0.028 / 0.850 (measured) | 0.035 / 0.767 (measured) | — / 0.000 (anchor_rejected) | 0.068 / 0.467 (measured) |
| C-clamp | a C-clamp | shap-e-text300m | 0.054 / 0.562 (measured) | 0.053 / 0.611 (measured) | 0.058 / 0.532 (measured) | 0.084 / 0.277 (measured) |
| elephant | an elephant | shap-e-text300m | 0.049 / 0.619 (measured) | 0.051 / 0.598 (measured) | 0.048 / 0.634 (measured) | 0.048 / 0.636 (measured) |

| generator | n | valid rate | mean CD-L1 (valid) | mean F@0.02 | mean F@0.05 (valid) | mean F@0.05 (failures = 0) |
|---|---|---|---|---|---|---|
| berkelium-textshape | 12 | 1.00 | 0.061 | 0.262 | 0.566 | 0.566 |
| shap-e-text300m | 12 | 1.00 | 0.047 | 0.372 | 0.669 | 0.669 |
| zimage-triposr | 12 | 0.50 | 0.047 | 0.332 | 0.327 | 0.327 |
| zimage-triposr-v2 | 12 | 1.00 | 0.055 | 0.344 | 0.603 | 0.603 |
| auto | 12 | 1.00 | 0.047 | 0.372 | 0.669 | 0.669 |

- shap-e-text300m_better_chamfer_than_native: 7/12
- zimage-triposr_better_chamfer_than_native: 4/6
- zimage-triposr-v2_better_chamfer_than_native: 6/12
- zimage-triposr_vs_shap_e: better Chamfer on 3/6 objects; paired ΔCD -0.000 (95% CI [-0.009, 0.0083]); paired ΔF@0.05 -0.342 (95% CI [-0.5808, -0.1124]).
- zimage-triposr-v2_vs_shap_e: better Chamfer on 5/12 objects; paired ΔCD 0.008 (95% CI [-0.0013, 0.017]); paired ΔF@0.05 -0.066 (95% CI [-0.1567, 0.0221]).

### B1 seed variance — shap-e-text300m (seeds 0-2)

Objects with all 3 seeds: 12/12; mean within-object std of Chamfer-L1: 0.0064; mean over seeds and objects: 0.050.

| prompt | seed 0 CD (dev) | seed 1 CD (dev) | seed 2 CD (dev) | std |
|---|---|---|---|---|
| a blue coffee mug | 0.044 (cpu) | 0.042 (cuda) | 0.038 (cuda) | 0.0026 |
| a white porcelain teapot | 0.122 (cpu) | 0.108 (cuda) | 0.136 (cuda) | 0.0118 |
| a white porcelain pitcher | 0.028 (cpu) | 0.031 (cuda) | 0.054 (cuda) | 0.0117 |
| a deep bowl | 0.015 (cpu) | 0.017 (cuda) | 0.019 (cuda) | 0.0015 |
| a teal plastic easter basket | 0.040 (cpu) | 0.041 (cuda) | 0.079 (cuda) | 0.0179 |
| a men's boot | 0.034 (cpu) | 0.030 (cuda) | 0.021 (cuda) | 0.0052 |
| a boat shoe | 0.020 (cpu) | 0.018 (cuda) | 0.018 (cuda) | 0.0009 |
| a school bus | 0.045 (cpu) | 0.055 (cuda) | 0.057 (cuda) | 0.0052 |
| a turboprop airplane with a pilot | 0.072 (cpu) | 0.071 (cuda) | 0.067 (cuda) | 0.0022 |
| a beige teddy bear | 0.035 (cpu) | 0.036 (cuda) | 0.038 (cuda) | 0.0013 |
| a C-clamp | 0.053 (cpu) | 0.055 (cuda) | 0.083 (cuda) | 0.0134 |
| an elephant | 0.051 (cuda) | 0.058 (cuda) | 0.057 (cuda) | 0.0032 |

### B1 seed variance — zimage-triposr (seeds 0-2)

Objects with all 3 seeds: 3/12; mean within-object std of Chamfer-L1: 0.0031; mean over seeds and objects: 0.052.

| prompt | seed 0 CD (dev) | seed 1 CD (dev) | seed 2 CD (dev) | std |
|---|---|---|---|---|
| a blue coffee mug | 0.062 (cuda) | — (cuda) | — (cuda) | — |
| a white porcelain teapot | — (cuda) | — (cuda) | — (cuda) | — |
| a white porcelain pitcher | — (cuda) | — (cuda) | — (cuda) | — |
| a deep bowl | — (cuda) | — (cuda) | — (cuda) | — |
| a teal plastic easter basket | — (cuda) | — (cuda) | — (cuda) | — |
| a men's boot | — (cuda) | — (cuda) | — (cuda) | — |
| a boat shoe | 0.020 (cuda) | — (cuda) | — (cuda) | — |
| a school bus | 0.027 (cuda) | 0.033 (cuda) | 0.041 (cuda) | 0.0059 |
| a turboprop airplane with a pilot | 0.068 (cuda) | 0.069 (cuda) | 0.076 (cuda) | 0.0038 |
| a beige teddy bear | — (cuda) | 0.068 (cuda) | — (cuda) | — |
| a C-clamp | 0.058 (cuda) | — (cuda) | 0.060 (cuda) | 0.0013 |
| an elephant | 0.048 (cuda) | 0.048 (cuda) | 0.051 (cuda) | 0.0014 |

### B1 seed variance — zimage-triposr-v2 (seeds 0-2)

Objects with all 3 seeds: 12/12; mean within-object std of Chamfer-L1: 0.0036; mean over seeds and objects: 0.052.

| prompt | seed 0 CD (dev) | seed 1 CD (dev) | seed 2 CD (dev) | std |
|---|---|---|---|---|
| a blue coffee mug | 0.056 (cuda) | 0.050 (cuda) | 0.050 (cuda) | 0.0025 |
| a white porcelain teapot | 0.116 (cuda) | 0.110 (cuda) | 0.112 (cuda) | 0.0025 |
| a white porcelain pitcher | 0.045 (cuda) | 0.032 (cuda) | 0.029 (cuda) | 0.0067 |
| a deep bowl | 0.034 (cuda) | 0.038 (cuda) | 0.034 (cuda) | 0.0021 |
| a teal plastic easter basket | 0.063 (cuda) | 0.059 (cuda) | 0.063 (cuda) | 0.0018 |
| a men's boot | 0.015 (cuda) | 0.021 (cuda) | 0.017 (cuda) | 0.0023 |
| a boat shoe | 0.017 (cuda) | 0.018 (cuda) | 0.019 (cuda) | 0.0008 |
| a school bus | 0.030 (cuda) | 0.035 (cuda) | 0.033 (cuda) | 0.0019 |
| a turboprop airplane with a pilot | 0.079 (cuda) | 0.076 (cuda) | 0.075 (cuda) | 0.0017 |
| a beige teddy bear | 0.068 (cuda) | 0.060 (cuda) | 0.059 (cuda) | 0.0041 |
| a C-clamp | 0.084 (cuda) | 0.049 (cuda) | 0.078 (cuda) | 0.0154 |
| an elephant | 0.048 (cuda) | 0.049 (cuda) | 0.051 (cuda) | 0.0012 |

## B2 — semantic relevance (CLIP R-precision)

Evaluator: OpenAI CLIP ViT-B/32 (not ViT-L/14, which conditions Shap-E) (sha256 `40d365715913c9da…`), 4 views per mesh, ranked against all 40 benchmark prompts.

| candidate | n | R@1 | R@5 |
|---|---|---|---|
| gt_scan | 12 | 1.00 | 1.00 |
| berkelium-textshape | 20 | 0.05 | 0.05 |
| shap-e-text300m | 20 | 0.95 | 1.00 |
| zimage-triposr | 20 | 0.35 | 0.50 |
| zimage-triposr-v2 | 20 | 0.85 | 1.00 |
| auto | 20 | 0.95 | 1.00 |

Per-prompt rank of the true prompt (1 = best):

| prompt | gt_scan | berkelium-textshape | shap-e-text300m | zimage-triposr | zimage-triposr-v2 | auto |
|---|---|---|---|---|---|---|
| a blue coffee mug | 1 | 13 | 1 | 1 | 1 | 1 |
| a white porcelain teapot | 1 | 17 | 2 | 29 (anchor_rejected) | 1 | 2 |
| a white porcelain pitcher | 1 | 29 | 1 | 33 (anchor_rejected) | 1 | 1 |
| a deep bowl | 1 | 6 | 1 | 19 (anchor_rejected) | 2 | 1 |
| a teal plastic easter basket | 1 | 32 | 1 | 37 (anchor_rejected) | 5 | 1 |
| a men's boot | 1 | 6 | 1 | 11 (anchor_rejected) | 1 | 1 |
| a boat shoe | 1 | 15 | 1 | 1 | 1 | 1 |
| a school bus | 1 | 7 | 1 | 1 | 1 | 1 |
| a turboprop airplane with a pilot | 1 | 39 | 1 | 1 | 1 | 1 |
| a beige teddy bear | 1 | 33 | 1 | 18 (anchor_rejected) | 1 | 1 |
| a C-clamp | 1 | 16 | 1 | 1 | 2 | 1 |
| an elephant | 1 | 12 | 1 | 1 | 1 | 1 |
| a bicycle | — | 14 | 1 | 5 (anchor_rejected) | 1 | 1 |
| a ladder | — | 13 | 1 | 2 (anchor_rejected) | 1 | 1 |
| a hammer | — | 1 | 1 | 4 (anchor_rejected) | 1 | 1 |
| a watering can | — | 17 | 1 | 23 (anchor_rejected) | 1 | 1 |
| a flower vase | — | 8 | 1 | 20 (anchor_rejected) | 1 | 1 |
| an open cardboard box | — | 27 | 1 | 8 (anchor_rejected) | 1 | 1 |
| a glass wine bottle | — | 26 | 1 | 36 (anchor_rejected) | 1 | 1 |
| a shiny metal trophy | — | 13 | 1 | 1 | 1 | 1 |

Residual bias: evaluator (ViT-B/32) and Shap-E's conditioner (ViT-L/14) are both OpenAI CLIP models trained on the same WIT-400M data: Shap-E outputs may score higher partly because they were optimised toward a related embedding space, not only because they look right. zimage-triposr is conditioned by Qwen3 (a different lineage) and is not favoured this way.

## B3 — retention on procedural prompts (existing protocol berkelium-text3d-eval/1)

| generator | n | success | mean CD-L1 | mean F@0.05 | part-count acc | colour acc |
|---|---|---|---|---|---|---|
| berkelium-textshape | 12 | 1.00 | 0.094 | 0.356 | 0.42 | 0.96 |
| auto | 12 | 1.00 | 0.094 | 0.356 | 0.42 | 0.96 |
| shap-e-text300m | 12 | 1.00 | 0.185 | 0.184 | 0.00 | 0.38 |
| zimage-triposr | 12 | 0.83 | 0.167 | 0.174 | 0.25 | 0.17 |
| zimage-triposr-v2 | 12 | 1.00 | 0.142 | 0.199 | 0.08 | 0.42 |

Routes: berkelium-textshape.

## B4 — composition: generators × {single-shot, planner}

CD = Chamfer-L1 in the shared normalized frame (existing protocol, no alignment); aligned CD = B1 alignment; part count = existing rule (connected components == GT); touch-merged = components after merging pieces within 0.02 (evaluate.py's 'touching parts merge'), same for GT.

| candidate | n | success | part-count acc | touch-merged acc | CD-L1 | aligned CD-L1 | colour acc |
|---|---|---|---|---|---|---|---|
| berkelium-textshape:single | 8 | 1.00 | 0.75 | 0.75 | 0.085 | 0.046 | 0.94 |
| berkelium-textshape:planner | 8 | 1.00 | 0.38 | 0.38 | 0.048 | 0.027 | 1.00 |
| shap-e-text300m:single | 8 | 1.00 | 0.12 | 0.12 | 0.185 | 0.087 | 0.38 |
| shap-e-text300m:planner | 8 | 1.00 | 0.00 | 0.00 | 0.059 | 0.033 | 0.50 |
| zimage-triposr:single | 8 | 0.50 | 0.12 | 0.00 | 0.197 | 0.066 | 0.00 |
| zimage-triposr:planner | 8 | 0.25 | 0.12 | 0.12 | 0.100 | 0.060 | 0.00 |
| zimage-triposr-v2:single | 8 | 1.00 | 0.38 | 0.38 | 0.149 | 0.061 | 0.12 |
| zimage-triposr-v2:planner | 8 | 1.00 | 0.50 | 0.38 | 0.093 | 0.055 | 0.38 |

| asset | caption | GT comps | berkelium-textshape:single | berkelium-textshape:planner | shap-e-text300m:single | shap-e-text300m:planner | zimage-triposr:single | zimage-triposr:planner | zimage-triposr-v2:single | zimage-triposr-v2:planner |
|---|---|---|---|---|---|---|---|---|---|---|
| next_to__box_box__003 | a large green box to the left of a purple box | 2 | 2c / CD 0.072 | 5c / CD 0.041 | 1c / CD 0.237 | 93c / CD 0.072 | Nonec / CD — | Nonec / CD — | 2c / CD 0.102 | 2c / CD 0.092 |
| next_to__cylinder_cone__003 | a small green cylinder beside a cyan cone | 2 | 2c / CD 0.084 | 3c / CD 0.046 | 3c / CD 0.195 | 3c / CD 0.090 | 1c / CD 0.285 | 2c / CD 0.087 | 1c / CD 0.226 | 2c / CD 0.070 |
| next_to__sphere_cone__003 | a blue sphere next to a small yellow cone | 2 | 1c / CD 0.099 | 2c / CD 0.021 | 2c / CD 0.263 | 4c / CD 0.081 | 26c / CD 0.166 | 6c / CD 0.114 | 84c / CD 0.171 | 29c / CD 0.053 |
| next_to__torus_torus__003 | a large white torus next to a large cyan torus | 2 | 2c / CD 0.062 | 5c / CD 0.052 | 15c / CD 0.146 | 19c / CD 0.023 | Nonec / CD — | Nonec / CD — | 2c / CD 0.291 | 2c / CD 0.093 |
| on_top_of__cylinder_sphere__003 | a large orange cylinder on top of a small yellow sphere | 1 | 1c / CD 0.051 | 2c / CD 0.031 | 10c / CD 0.125 | 29c / CD 0.051 | Nonec / CD — | Nonec / CD — | 1c / CD 0.070 | 2c / CD 0.043 |
| on_top_of__sphere_sphere__003 | a large red sphere on top of a large cyan sphere | 1 | 1c / CD 0.041 | 2c / CD 0.029 | 13c / CD 0.220 | 28c / CD 0.086 | 4c / CD 0.115 | Nonec / CD — | 6c / CD 0.070 | 38c / CD 0.037 |
| on_top_of__sphere_torus__003 | a large green sphere on top of a large yellow torus | 2 | 3c / CD 0.175 | 2c / CD 0.093 | 18c / CD 0.182 | 10c / CD 0.029 | Nonec / CD — | Nonec / CD — | 1c / CD 0.119 | 2c / CD 0.201 |
| on_top_of__torus_sphere__003 | a small purple sphere with a white torus on top | 2 | 2c / CD 0.098 | 2c / CD 0.074 | 37c / CD 0.108 | 29c / CD 0.036 | 2c / CD 0.222 | Nonec / CD — | 23c / CD 0.144 | 11c / CD 0.154 |

## Router

Rule: native iff every content word is in its vocabulary; otherwise the first available of shap-e-text300m > zimage-triposr (E-T3 evidence order, D-029). Domain accuracy (B1/B2 → an open-world generator, B3/B4 → native): 40/40. Agreement with the per-prompt best generator on B1 (Chamfer oracle over berkelium-textshape, shap-e-text300m, zimage-triposr, zimage-triposr-v2): 5/12. the open-world default was chosen from these same B1/B2 numbers (NEURAL_RENDERING_DECISIONS.md); auto's B1/B2 scores are therefore in-sample, not an independent test of the router

