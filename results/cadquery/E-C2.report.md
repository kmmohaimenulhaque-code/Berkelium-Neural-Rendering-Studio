# E-C2 — text → CadQuery compiler (Qwen/Qwen3-32B + LoRA)

Run 2026-10-18T11:18:17Z on `AMD Radeon Graphics` (backend transformers). Decision rule D-034 (pre-registered in `NEURAL_RENDERING_DECISIONS.md`). Every number below is in `E-C2.json`; per-item records are at `/root/berkelium_run/eval/E-C2_items.jsonl.gz` (sha256 `7511d9e0af2c7232…`).

| group | n | base exec+valid | tuned exec+valid | base IoU | tuned IoU | tuned exec+valid (≤2 repairs) |
|---|---|---|---|---|---|---|
| all_heldout | 400 | 0.000 | 0.863 | 0.000 | 0.442 | 0.890 |
| test | 300 | 0.000 | 0.887 | 0.000 | 0.549 | 0.893 |
| heldout_families | 100 | 0.000 | 0.790 | 0.000 | 0.121 | 0.880 |
| test:cad_recode | 100 | 0.000 | 0.760 | 0.000 | 0.340 | 0.770 |
| test:families | 100 | 0.000 | 0.920 | 0.000 | 0.640 | 0.930 |
| test:text2cadquery | 100 | 0.000 | 0.980 | 0.000 | 0.667 | 0.980 |

## D-034

| criterion | value | threshold | pass |
|---|---|---|---|
| execute+valid (held-out, first attempt) | 0.8625 | >= 0.85 | PASS |
| IoU tuned − base, paired 95 % CI | 0.4420 [0.4005, 0.4848] | CI lower > 0 | PASS |
| held-out-family execute | 0.8 | >= 0.6 | PASS |
| sandbox rejects injected unsafe scripts | 1.0000 (42 scripts) | 1.0 | PASS |

**Verdict: PASS** — promote to Studio's default for engineering prompts

Evidence class: ROCm/AMD. Sandbox network isolation: netns.
