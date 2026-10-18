# BerkeliumGPT (local): hand-off

- **Ollama (GGUF Q4_K_M, 19.8 GB):** https://huggingface.co/Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-GGUF  ->  `ollama run hf.co/Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-GGUF:Q4_K_M`
- **Full merged weights (bf16 safetensors, 65.5 GB):** https://huggingface.co/Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged
- **LoRA adapter only (2.15 GB):** https://huggingface.co/Mhaquehaque/berkelium-qwen3-32b-cadquery-v1
- **Base model:** `Qwen/Qwen3-32B` @ `9216db5781bf21249d130ec9da846c4624c16137`; trained 2780 / 2780 steps
- **Type:** the GGUF and `-merged` repos are full weights (base + LoRA merged); the adapter repo is LoRA only.
- **GGUF sha256:** `e432ceb3922d6dc70d7890f1fefef5666528ec6278cb46f04c033f44ae2dc777`
- **Eval:** D-034 **NOT EVALUATED**: opt-in only; the E-C2 eval was skipped (deadline), D-034 not applied. The Q4_K_M GGUF itself was not evaluated.
- **Use:** system prompt is in the repo (`system`); temperature 0; run the returned script in a sandbox.
- **Licence:** non-commercial (CAD-Recode CC-BY-NC-4.0 in the training data).

(Copied from the droplet's `results/cadquery/HANDOFF_TEAMMATE.md`, written by `scripts/cq_gguf.sh` at 2026-10-18T08:25:21Z.)
