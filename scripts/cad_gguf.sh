#!/usr/bin/env bash
# Merge the E-C1 LoRA into Qwen3-8B (fp16, CPU) and export GGUF Q4_K_M with llama.cpp (pinned). Runs on a Kaggle
# CPU kernel; the adapter comes from the training kernel's verified output (scripts/cad_fetch_inputs.py).
set -euo pipefail
LLAMA_CPP_COMMIT=9d1878a4354e743529fbe09e7350a957f5e39e9a
BASE=Qwen/Qwen3-8B; REV=b968826d9c46dd6066d109eabc6255188de91218
OUT=exports/gguf; W=/tmp/gguf_work; mkdir -p "$OUT" "$W"
python - <<'PY'
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, rev = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
m = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.float16, low_cpu_mem_usage=True)
m = PeftModel.from_pretrained(m, "exports/qwen3-8b-cad-v1/adapter").merge_and_unload()
m.save_pretrained("/tmp/gguf_work/merged", safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained("exports/qwen3-8b-cad-v1/adapter").save_pretrained("/tmp/gguf_work/merged")
print("merged")
PY
git clone -q https://github.com/ggml-org/llama.cpp "$W/llama.cpp" && git -C "$W/llama.cpp" checkout -q "$LLAMA_CPP_COMMIT"
pip install -q -r "$W/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" || pip install -q gguf sentencepiece
python "$W/llama.cpp/convert_hf_to_gguf.py" "$W/merged" --outtype f16 --outfile "$W/f16.gguf"
rm -rf "$W/merged"
cmake -S "$W/llama.cpp" -B "$W/build" -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$W/build" --target llama-quantize -j"$(nproc)" >/dev/null
"$W/build/bin/llama-quantize" "$W/f16.gguf" "$OUT/berkelium-qwen3-8b-cad-v1.Q4_K_M.gguf" Q4_K_M
ls -la "$OUT"
