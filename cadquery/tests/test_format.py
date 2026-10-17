import json

import cqformat


def test_render_parse_round_trip():
    code = "import cadquery as cq\nresult = cq.Workplane().box(1, 2, 3)"
    s = cqformat.render({"summary": "box", "assumptions": ["a"], "open_questions": []}, code)
    h, c, problems = cqformat.parse(s)
    assert problems == [] and h["units"] == "mm" and c.strip() == code
    assert json.loads(s.split("\n", 1)[0])["summary"] == "box"


def test_parse_reports_problems():
    h, c, p = cqformat.parse("<think>\n</think>\nno header\n```python\nresult = 1\n```")
    assert h is None and c.strip() == "result = 1" and "missing JSON header line" in p
    h, c, p = cqformat.parse('{"summary": "x"}\n```python\nresult = 1')
    assert c is not None and any("unterminated" in x for x in p) and any("missing" in x for x in p)


def test_repair_messages_shape():
    m = cqformat.repair_messages("a box", "bad", "NameError: x")
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"] and "NameError" in m[-1]["content"]
