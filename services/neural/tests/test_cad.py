"""E-C1 design compiler path: JSON extraction, family generators (deterministic, unit conversion, requirements),
and the Studio ``cad`` generator with a stub compiler, realised by the Berkelium core. No GPU, no model weights."""

import json

import numpy as np
import pytest

from berkelium_neural.cad import families as FAM
from berkelium_neural.cad.prompts import compact, extract_json, messages, system_prompt


def test_extract_json_tolerates_think_fences_and_braces_in_strings():
    obj = {"a": "x}{y\\\"", "b": [1, {"c": 2}]}
    txt = "<think>\nplan {not json}\n</think>\n```json\n" + compact(obj) + "\n```\ntrailing"
    assert extract_json(txt) == obj
    assert extract_json("no json here") is None


def test_prompt_states_the_unit_rule_and_catalogue():
    s = system_prompt("- gear.spur_pair@0.1 (cem): ...")
    assert "H+2[mm]" in s and "gear.spur_pair@0.1" in s and "open_questions" in s
    m = messages("a washer", "-", {"k": 1})
    assert [x["role"] for x in m] == ["system", "user", "assistant"] and m[2]["content"] == '{"k":1}'


def test_families_are_deterministic_and_cover_heldout():
    assert set(FAM.HELDOUT_FAMILIES) <= set(FAM.FAMILIES) and len(FAM.FAMILIES) >= 30
    for fam in FAM.FAMILIES:
        a, b = FAM.sample(fam, 7, False), FAM.sample(fam, 7, False)
        assert a == b


def test_precise_text_states_sizes_and_requirements_use_the_users_unit():
    for seed in range(40):
        r = FAM.sample("plate_holes", seed, False)
        if r is None:
            continue
        text, prop = r
        reqs = {q["quantity"]: q for q in prop["specification"]["requirements"]}
        L = next(p for p in prop["specification"]["parameters"] if p["id"] == "L")
        unit = reqs["body.extent_x"]["target"]["unit"]
        assert f"{reqs['body.extent_x']['target']['value']:g}" in text.replace(".0 ", " ") or unit in text
        assert L["unit"] == "mm" and np.isclose(L["value"], reqs["body.extent_x"]["target"]["value"] *
                                                FAM.UNIT_MM[unit], rtol=1e-6)


def test_vague_text_has_no_numbers_and_raises_an_open_question():
    text, prop = FAM.sample("flange", 3, True)
    assert not any(ch.isdigit() for ch in text.split(",")[0])
    assert prop["intent"]["open_questions"] and prop["intent"]["assumptions"]
    assert prop["specification"]["requirements"] == []


berkelium = pytest.importorskip("berkelium", reason="Berkelium core not installed (configs/environments/berkelium-core.txt)")


@pytest.mark.parametrize("fam", ["plate_holes", "rocket_nozzle", "pressure_vessel", "custom_part"])
def test_family_targets_realise_in_the_core(fam):
    from berkelium.pipeline import run
    text, prop = FAM.sample(fam, 11, False)
    rec = run(prop, realize=True, backend="manifold").record
    assert rec is not None and rec.evaluation.validation.summary in ("pass", "warn")


def test_cad_generator_with_a_stub_compiler_returns_core_geometry():
    from berkelium_neural.cad.generator import CadGenerator
    text, prop = FAM.sample("washer", 5, False)
    seen = []
    g = CadGenerator(compiler=lambda t: seen.append(t) or "<think></think>" + json.dumps(prop))
    out = g.generate(text)
    rep = out["report"]
    assert seen == [text] and rep["source"] == "injected" and rep["valid"]
    assert rep["validation"]["summary"] in ("pass", "warn") and rep["proposal"] == prop
    v = out["vertices"]
    assert v.dtype == np.float32 and np.isclose(np.ptp(v, 0).max(), 1.0, atol=1e-5) and len(out["faces"]) > 10
    od = next(p["value"] for p in prop["specification"]["parameters"] if p["id"] == "OD")
    assert np.isclose(rep["scale_mm_per_unit"], od, rtol=1e-3)            # washer: the largest extent is its OD


def test_cad_generator_accepts_a_proposal_json_prompt_without_a_model():
    from berkelium_neural.cad.generator import CadGenerator
    _, prop = FAM.sample("hex_nut", 2, False)
    out = CadGenerator().generate(json.dumps(prop))
    assert out["report"]["source"] == "proposal_json" and out["report"]["completion"] is None


def test_cad_generator_refuses_invalid_proposals_with_the_core_reason():
    from berkelium_neural.cad.generator import CadGenerator, CadUnavailable
    with pytest.raises(CadUnavailable, match="schema"):
        CadGenerator(compiler=lambda t: '{"kind": "design_proposal", "structure": {"components": []}}').generate("x")
