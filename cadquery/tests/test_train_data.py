import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class Tok:
    """Character-level stand-in for a chat tokenizer."""
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert enable_thinking is False
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) % 250 + 1 for c in s]}


def test_encode_masks_prompt_and_pack_fills_rows():
    tr = load("cq_train")
    row = {"text": "a 10 mm cube", "completion": '{"summary":"cube"}\n```python\nresult = 1\n```', "kind": "gen"}
    e = tr.encode(Tok(), row, 4096)
    n_c = len(Tok()(row["completion"] + "<|im_end|>\n")["input_ids"])
    assert e["labels"][-n_c:] == e["input_ids"][-n_c:] and all(x == -100 for x in e["labels"][:-n_c])
    assert tr.encode(Tok(), row, 10) is None
    rep = dict(row, kind="repair", broken_completion="bad", error="NameError")
    assert len(tr.encode(Tok(), rep, 4096)["input_ids"]) > len(e["input_ids"])
    rows = tr.pack([e] * 7, 3 * len(e["input_ids"]) + 5, 0, seed=0)
    assert len(rows) == 3 and all(len(r["input_ids"]) == 3 * len(e["input_ids"]) + 5 for r in rows)
    assert rows[0]["position_ids"][len(e["input_ids"])] == 0 and sum(r["n_tokens"] for r in rows) == 7 * len(e["input_ids"])


def test_data_helpers():
    d = load("cq_data")
    code = "import cadquery as cq\nfrom cadquery import exporters\nfrom cadquery.vis import show\n" \
           "assembly = cq.Workplane().box(1, 1, 1)\nshow(assembly)\ncq.exporters.export(assembly, 'x.stl')\n"
    c = d.clean_script(code)
    assert "exporters" not in c and "show" not in c and c.rstrip().endswith("result = assembly")
    import random
    import sandbox
    good = "import cadquery as cq\nw = 2\nresult = cq.Workplane().box(w, 1, 1).translate((1, 0, 0))\n"
    kinds = set()
    for s in range(40):
        b = d.break_script(good, random.Random(s))
        assert b is not None and b[0] != good
        kinds.add(b[1])
    assert kinds == {"name", "method", "result", "syntax", "import"}
    assert sandbox.check_script(d.break_script(good, random.Random(0))[0]) is not None
    assert d.norm_code("a = 1  # x\n\n") == "a = 1"
