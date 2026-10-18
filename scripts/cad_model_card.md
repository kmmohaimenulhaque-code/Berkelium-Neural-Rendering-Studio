---
license: apache-2.0
base_model: Qwen/Qwen3-8B
library_name: peft
pipeline_tag: text-generation
tags: [lora, cad, engineering, text-to-cad, berkelium, qwen3]
---

# berkelium-qwen3-8b-cad-v1: text → engineering design (opt-in beta)

A LoRA adapter for **Qwen/Qwen3-8B** (rev `b968826d9c46dd6066d109eabc6255188de91218`). It turns a plain-language
engineering request into a **Berkelium `DesignProposal` JSON**: a parametric design with named dimensions (converted to
mm), checkable size requirements, assumptions and open questions. It also includes either a procedural Geometry IR op graph
(profiles, extrude/revolve, booleans, patterns) or a call to a catalogued engineering model (spur-gear pair, cantilever beam).

The model only **proposes**. The deterministic [Berkelium core](https://github.com/kmmohaimenulhaque-code/BERKELIUM-NEURAL-RENDERING-AND-SCIENCES)
builds the exact geometry (OCCT STEP / Manifold mesh), measures it and validates it. In Berkelium Studio the result is then
fitted with Gaussians and splat-rendered in the viewport. Without the core, the output is just JSON.

GGUF (Q4_K_M, merged, 5.03 GB) for Ollama / llama.cpp: [Mhaquehaque/berkelium-qwen3-8b-cad-v1-GGUF](https://huggingface.co/Mhaquehaque/berkelium-qwen3-8b-cad-v1-GGUF).

## What it does well
On held-out prompts for the object families it was trained on, the core accepts **99.8 %** of its designs, **99.7 %**
meet every size the prompt states, and the mean volume IoU against the reference is **0.94**. Base Qwen3-8B scores
**0 %** on the same prompts: it never produces a valid DesignProposal. Units (mm, cm, in), counts and hole
patterns are carried into the design.

Real prompts from the held-out test set that it reproduced exactly (IoU ≥ 0.98, every requirement met):
- *"Draw up a corner bracket with legs of 40 mm and 117 mm, 5 mm thick, 93 mm wide, with a mounting hole in each leg."*
- *"Make a pipe flange and 3 bolt holes, with a 53 mm bore, 290 mm outer diameter, 22 mm thick."*
- *"Draw up a flat plate 26.3 cm by 10.4 cm, 0.37 cm thick, with a 5 x 4 grid of 1.7 cm holes."*
- *"A spindle with 2 sections: 6 cm diameter x 13.6 cm; 2.8 cm diameter x 7.2 cm."*
- *"I need a nut 0.95 in across flats, 0.7 in thick."*
- *"Make a tee junction 113 mm OD, 55 mm ID, run length 432 mm, branch 190 mm from the centreline."*
- *"Design a wheel 9 spokes, 159 mm across, 62 mm wide rim."*
- *"Design an electronics enclosure base 252 mm x 34 mm x 112 mm, with 5.5 mm walls."*
- *"Generate a spur gearset module 2.5 mm, with a 3.22581:1 reduction."* (catalogued gear model, 14/14 correct)

**Strong families (realise ≈ 1.0, IoU ≥ 0.9):** plates with hole grids, L-brackets, U-channels, I-beams, flanges,
washers, bushings, stepped shafts, hex nuts/bolts (unthreaded), pulleys, spoked wheels, pillow blocks, toothed discs,
heat sinks, open enclosures, pipe tees, propellers, mugs, bottles, vases, tables, ladders, rocket airframes (ogive + fins +
bell), the lantern, spur-gear pairs, cantilever beams, and the "base block/disc + holes / pockets / bosses / slots /
counterbores / bolt circle" feature grammar. Vague prompts ("a small bracket") get typical sizes, listed as
assumptions plus an open question.

## Limitations (read before using)
- **Unseen object kinds:** on four families held out of training (pipe elbow, chair, knob, hinge leaf) only 49 %
  of designs realise and the mean IoU is 0.06. It imitates the trained families rather than reasoning about new
  shapes. It is **not** a "make anything" model.
- **Weaker trained families:** rocket nozzle contour (IoU 0.25: the bell shape is often wrong even when it builds),
  honeycomb panels (0.50), pressure vessels (0.78), trusses (0.82).
- **The geometry language is limited:** no fillets, chamfers, threads, sweeps along paths or lofts. Such requests are
  approximated.
- **Template-generated training text:** phrasings far from "a <part>, <dimensions>" may degrade results. Real user
  wording was not tested.
- **The GGUF is a 4-bit quantisation of the merged model.** The evaluation above used the NF4 adapter, and the GGUF
  output may differ slightly.
- **No engineering verdicts:** the model never computes stresses or checks fitness for use. Only the Berkelium
  core's validation (and FEM, where configured) can do that.

## Evaluation (E-C1, scored by the Berkelium core; rule D-033 pre-registered before training)
| model | set | n | schema-valid | realises | stated sizes met | IoU |
|---|---|---|---|---|---|---|
| zero-shot Qwen3-8B | test (trained families, unseen seeds) | 451 | 0.00 | 0.00 | 0.00 | 0.00 |
| **this adapter** | test | 451 | **1.00** | **0.998** | **0.997** | **0.936** |
| zero-shot Qwen3-8B | held-out families | 354 | 0.00 | 0.00 | 0.00 | 0.00 |
| **this adapter** | held-out families | 354 | 0.63 | 0.49 | 0.15 | 0.056 |
| **this adapter** | spur-gear model | 14 | 1.00 | 1.00 | 1.00 | 1.00 |

**D-033: 5 of 6 checks pass. Held-out-family realise is 0.49, below the required 0.60, so it is not promoted.** It is offered as an
opt-in "Engineering CAD" generator, not the default router target. NF4 on Kaggle T4, greedy decoding, thinking off.

## Training
QLoRA (NF4 base, fp16 compute, fp32 LoRA r=32 / α 64 on all linear layers), DDP on 2× Kaggle T4. Data:
`berkelium-cad-v1` (18,712 core-verified rows; a random 12,000-row subset, 15.5M tokens). 601 steps (0.80 epoch,
stopped by a 10 h guard), train loss 1.03 → 0.005. Data manifest sha256 `bb07a39a…`, adapter sha256 `c76b64ee…`.

## Use
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")
model = PeftModel.from_pretrained(base, "Mhaquehaque/berkelium-qwen3-8b-cad-v1")
```
Use the system prompt `cad-0.1` (`services/neural/berkelium_neural/cad/prompts.py`) with thinking off, then run the JSON through
`berkelium.pipeline.run(...)`. In Berkelium Studio, set `BERKELIUM_CAD_ADAPTER=Mhaquehaque/berkelium-qwen3-8b-cad-v1` (CUDA) or point
`BERKELIUM_CAD_URL` at an Ollama/vLLM endpoint, then choose **Engineering CAD** in Generate.

Code and reports: github.com/kmmohaimenulhaque-code/Berkelium-Neural-Rendering-Studio (branch
`Berkelium-Neural-Rendering-Studio-neural-v1`, `results/cad/E-C1/`).
