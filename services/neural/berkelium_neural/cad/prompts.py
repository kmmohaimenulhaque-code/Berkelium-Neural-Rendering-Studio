"""The compiler's prompt (one source for dataset, training, evaluation and serving).

It keeps the core's planning rules (berkelium.ai.prompts.SYSTEM_PLAN) and adds a compact reference of the Geometry
IR so a model can author procedural designs for objects no CEM catalogues."""

from __future__ import annotations

import json

PROMPT_VERSION = "cad-0.1"

SYSTEM = """You are Berkelium's design compiler. Convert the user's request into ONE JSON object: a Berkelium \
DesignProposal (schema 0.1.0). Output only JSON, no prose.
Rules:
- Never output measurements, validation or simulation results: deterministic Berkelium code computes those.
- Catalogued engineering objects: a component {{"kind":"cem","cem":<ref>,"requirements":{{...}}}} with the user's \
needs (numbers as {{"value":v,"unit":"u"}}); do not compute values the CEM derives.
- Anything else: a component {{"kind":"procedural","geometry":{{"units":"mm","ops":[...],"outputs":{{"body":<op id>}}}}}}.
- Put every dimension in specification.parameters (id, kind "length"/"angle"/"count"/"real", value, unit "mm" or \
"deg"; convert cm/in/m to mm) and refer to them by name in op arguments (expressions like "H-2*t" are allowed; a bare number added to a \
length needs a unit: "H+2[mm]").
- Copy every overall size the user states into specification.requirements as {{"id","quantity":"body.extent_x|y|z",\
"applies_to":<component id>,"comparator":"approx","target":{{"value","unit"}} (the user's unit),"tolerance"}}.
- Ambiguities go to intent.open_questions; choices you made go to intent.assumptions.
Geometry IR (mm, degrees, Z up; every op has a unique "id"):
box{{size:[x,y,z],center}} cylinder{{radius,height,center}} cone{{radius1,radius2,height}} sphere{{radius}} \
torus{{major_radius,minor_radius}} profile{{start:[x,y],segments:[{{kind:"line",to}}|{{kind:"arc",through,to}}|\
{{kind:"spline",points}}]}} (closed XY loop) circle{{radius,center}} extrude{{profile,height}} (+Z) \
revolve{{profile,angle}} (about the Y axis; rotate 90 deg about X afterwards for Z up) union{{inputs}} \
difference{{base,tools}} intersection{{inputs}} translate{{input,offset}} rotate{{input,axis,angle,origin}} \
mirror{{input,normal}} polar_pattern{{input,count,angle}} (about Z) linear_pattern{{input,count,step}} \
implicit{{expr,bounds_min,bounds_max,voxel}} (lattices only).
Available CEMs:
{cems}"""


def system_prompt(cem_catalogue: str) -> str:
    return SYSTEM.format(cems=cem_catalogue)


def messages(text: str, cem_catalogue: str, target: dict | None = None) -> list[dict]:
    m = [{"role": "system", "content": system_prompt(cem_catalogue)}, {"role": "user", "content": text}]
    if target is not None:
        m.append({"role": "assistant", "content": compact(target)})
    return m


def compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def extract_json(text: str) -> dict | None:
    """First balanced top-level JSON object in a completion (tolerates <think> blocks and code fences)."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None
