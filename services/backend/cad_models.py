"""Team Berkelium's own CAD models, and how the backend talks to them.

These models are not chat models. Each one was fine-tuned with one exact system
prompt and with Qwen3 "thinking" off. Sending them the general chat prompt, or
earlier chat turns they never saw in training, makes their output worse. So the
backend recognises them by model name and sends their own prompt, the user's
latest message only, and greedy decoding.

Matching by name works wherever the model runs: the local Ollama app, an Ollama
server on a remote GPU machine, or an OpenAI-compatible server.
"""

from dataclasses import dataclass
from typing import Optional

# The 32B prompt is a copy of the "system" file in the model's GGUF repo
# (Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-GGUF). Sending it ourselves keeps
# the model correct on servers that ignore that file, such as vLLM.
QWEN3_32B_CADQUERY_PROMPT = """You are a CAD compiler. Turn the request into one line of JSON {"summary","units":"mm","assumptions":[],"open_questions":[]} followed by one ```python block: a CadQuery script in millimetres that imports only cadquery and math and ends with `result = ...`. Unstated sizes go in assumptions; missing key sizes go in open_questions. /no_think"""

# cad-0.1: the prompt the 8B model was trained on. Generated from
# services/neural/berkelium_neural/cad/prompts.py with the CEM catalogue of the
# Berkelium core pinned at 737dc35. Do not edit by hand: a changed prompt is a
# different model as far as the weights are concerned.
CAD_01_PROMPT = """You are Berkelium's design compiler. Convert the user's request into ONE JSON object: a Berkelium DesignProposal (schema 0.1.0). Output only JSON, no prose.
Rules:
- Never output measurements, validation or simulation results: deterministic Berkelium code computes those.
- Catalogued engineering objects: a component {"kind":"cem","cem":<ref>,"requirements":{...}} with the user's needs (numbers as {"value":v,"unit":"u"}); do not compute values the CEM derives.
- Anything else: a component {"kind":"procedural","geometry":{"units":"mm","ops":[...],"outputs":{"body":<op id>}}}.
- Put every dimension in specification.parameters (id, kind "length"/"angle"/"count"/"real", value, unit "mm" or "deg"; convert cm/in/m to mm) and refer to them by name in op arguments (expressions like "H-2*t" are allowed; a bare number added to a length needs a unit: "H+2[mm]").
- Copy every overall size the user states into specification.requirements as {"id","quantity":"body.extent_x|y|z","applies_to":<component id>,"comparator":"approx","target":{"value","unit"} (the user's unit),"tolerance"}.
- Ambiguities go to intent.open_questions; choices you made go to intent.assumptions.
Geometry IR (mm, degrees, Z up; every op has a unique "id"):
box{size:[x,y,z],center} cylinder{radius,height,center} cone{radius1,radius2,height} sphere{radius} torus{major_radius,minor_radius} profile{start:[x,y],segments:[{kind:"line",to}|{kind:"arc",through,to}|{kind:"spline",points}]} (closed XY loop) circle{radius,center} extrude{profile,height} (+Z) revolve{profile,angle} (about the Y axis; rotate 90 deg about X afterwards for Z up) union{inputs} difference{base,tools} intersection{inputs} translate{input,offset} rotate{input,axis,angle,origin} mirror{input,normal} polar_pattern{input,count,angle} (about Z) linear_pattern{input,count,step} implicit{expr,bounds_min,bounds_max,voxel} (lattices only).
Available CEMs:
- gear.spur_pair@0.1 (cem): External involute spur gear pair (20 deg full depth): sizing from ratio/module/centre distance or Lewis-based module selection; undercut, interference, contact-ratio and preliminary Lewis checks. requirements: ratio, ratio_tolerance, module[mm], center_distance[mm], face_width[mm], min_pinion_teeth, pinion_torque[N*mm], pinion_speed[rev/min], allowable_bending_stress[MPa], manufacturing, candidate_modules, bore_diameter_pinion[mm], bore_diameter_gear[mm]; required: ratio
- structure.cantilever_beam@0.1 (cem): Rectangular cantilever sized for an end load: stiffness + strength (Euler-Bernoulli), with a numerical 3-D elasticity check available as an L7 analysis requirements: tip_load_N, length_mm, max_tip_deflection_mm, allowable_stress_MPa, youngs_modulus_GPa, aspect_ratio; required: tip_load_N, length_mm, max_tip_deflection_mm, allowable_stress_MPa, youngs_modulus_GPa"""


@dataclass(frozen=True)
class CadModel:
    key: str      # lowercase text that appears in the model name
    label: str    # human name for logs
    system: str   # the exact system prompt the model was trained with


# Most specific first, so a future name that contains two keys picks the right one.
CAD_MODELS: tuple[CadModel, ...] = (
    CadModel("berkelium-qwen3-32b-cadquery", "Berkelium Qwen3 32B CadQuery", QWEN3_32B_CADQUERY_PROMPT),
    CadModel("berkelium-qwen3-8b-cad", "BerkeliumGPT 8B (cad-0.1)", CAD_01_PROMPT),
)

# Matches the training and evaluation setup: greedy, room for the prompt plus a
# full answer, and a cap so a runaway reply cannot hold the machine forever.
OLLAMA_OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 2048}

# Qwen3 ends every turn with this token. In raw mode Ollama applies no template,
# so it must be told where the answer stops.
QWEN3_STOP = ["<|im_end|>", "<|endoftext|>"]

# Loading a 20 GB model takes a while. Keep it in memory between questions.
OLLAMA_KEEP_ALIVE = "30m"


def find(model: Optional[str]) -> Optional[CadModel]:
    name = (model or "").lower()
    for cad in CAD_MODELS:
        if cad.key in name:
            return cad
    return None


def build_messages(cad: CadModel, message: str) -> list[dict]:
    # Single turn on purpose: the models were trained on one request each.
    return [{"role": "system", "content": cad.system}, {"role": "user", "content": message}]


def raw_prompt(cad: CadModel, message: str) -> str:
    """The exact text the models saw in training: Qwen3 chat format with thinking off.

    Ollama's think=false does not reliably do this. On Ollama 0.40.6 the 8B then
    writes a stray </think>, repeats its design, or reasons in prose and gets the
    geometry wrong. Writing the empty think block ourselves and sending the prompt
    raw removes the guesswork: one clean design, the same as the evaluation setup.
    """
    return (
        f"<|im_start|>system\n{cad.system}<|im_end|>\n"
        f"<|im_start|>user\n{message}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def clean_output(text: str) -> str:
    """Drop any reasoning a server still let through. Safety net for every route."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.replace("<think>", "").strip()
