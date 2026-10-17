---
license: apache-2.0
base_model: Qwen/Qwen3-8B
library_name: peft
pipeline_tag: text-generation
tags: [lora, cad, engineering, text-to-cad, berkelium, qwen3]
---

# berkelium-qwen3-8b-cad-v1 (EXPERIMENTAL, evaluation pending)

A LoRA adapter for **Qwen/Qwen3-8B** (revision `b968826d9c46dd6066d109eabc6255188de91218`). It turns a natural-language
engineering request into a **Berkelium `DesignProposal` JSON**: a parametric design with named parameters,
requirements, assumptions and open questions, plus a procedural Geometry IR op graph or a catalogued engineering model
(spur-gear pair, cantilever beam).

**The model only proposes.** Geometry, measurements and validation come from the deterministic Berkelium core
(OCCT exact B-rep / Manifold mesh, validation levels L0–L6). Without the core, the output is just JSON.

## Status
- **Not yet evaluated.** The held-out evaluation (E-C1: zero-shot Qwen3-8B vs this adapter, scored by the core,
  promotion rule D-033 pre-registered before training) is running. This card will be updated with the numbers.
- Treat it as experimental. The training text is template-generated, so the low validation loss below does
  **not** show generalisation.

## Training
| | |
|---|---|
| Method | QLoRA: NF4 base, fp16 compute, fp32 LoRA, r=32 / alpha 64 on all linear layers, DDP on 2× Kaggle T4 |
| Data | `berkelium-cad-v1`: 18,712 core-verified rows (33 part families + feature grammar + gear/beam CEMs + lantern); random 12,000-row training subset, 15.5M tokens |
| Steps | 601 (0.80 epoch, stopped by a 10 h time guard); 10.06 h |
| Loss | train 1.03 → 0.005; val 0.0042 (120 in-distribution rows) |
| Data manifest | sha256 `bb07a39a…fe49d` |
| Adapter | `adapter_model.safetensors` sha256 `c76b64ee…2f12f` (174.7 MB, fp16) |

Held-out families never trained on: pipe_elbow, chair, knob, hinge_leaf.

## Use
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")
model = PeftModel.from_pretrained(base, "Mhaquehaque/berkelium-qwen3-8b-cad-v1")
```
Use the system prompt from `berkelium_neural/cad/prompts.py` (prompt version `cad-0.1`), with thinking mode off.
Then run the JSON through `berkelium.pipeline.run(...)`. A GGUF (Q4_K_M) for Ollama/llama.cpp is published at
`Mhaquehaque/berkelium-qwen3-8b-cad-v1-GGUF`.

Code: github.com/enigmahacker/Berkelium-Neural-Rendering-Studio (branch `Berkelium-Neural-Rendering-Studio-neural-v1`).
