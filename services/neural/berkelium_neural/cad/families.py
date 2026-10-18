"""Parametric part families -> (natural-language request, DesignProposal) pairs for the E-C1 compiler dataset.

Every family authors procedural Geometry IR whose op arguments are expressions over named parameters, so a
proposal is an editable parametric design, not a frozen mesh. Texts come in two registers:

* precise: every stated size appears in the text in the user's unit (mm / cm / in); parameters are converted to mm
  and every overall size is copied into specification.requirements (body.extent_*) so the core can check it.
* vague: no numbers; the family's canonical size (scaled by "small"/"large") is used, listed in
  intent.assumptions, and the missing sizes are raised in intent.open_questions.

Unstated secondary dimensions are deterministic rules of the stated ones (recorded as assumptions), so a target
is a function of its text and therefore learnable. Nothing here is verified: scripts/cad_dataset.py runs every
proposal through the Berkelium core and keeps only those that realise and meet their requirements."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

UNIT_MM = {"mm": 1.0, "cm": 10.0, "in": 25.4}
SIZE_WORD = {0.6: ["small", "compact", "little", "mini"], 1.0: ["", "standard", "medium", "regular"],
             1.6: ["large", "big", "heavy", "oversized"]}
OPEN = ["", "Design ", "Model ", "Create ", "Make ", "I need ", "Generate ", "CAD model of ", "Please model ",
        "Build ", "Draw up "]


def _fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _a(word: str) -> str:
    w = word.lstrip()
    letter = len(w) > 1 and w[0].isupper() and w[1] in "- " or w[:2] in ("L-", "H-", "I-", "U-", "C-", "T-", "V-")
    vowel = w[:1] in "AEFHILMNORSX" if letter else (w[:1].lower() in "aeio" or w[:1].lower() == "u" and w[1:2] != "-")
    return ("an " if vowel else "a ") + word


class Builder:
    """Collects parameters, ops, requirements, assumptions and the text slots for one example."""

    def __init__(self, rng: np.random.Generator, vague: bool, comp: str = "part"):
        self.rng, self.vague, self.comp = rng, vague, comp
        self.unit = str(rng.choice(["mm"] * 6 + ["cm"] * 2 + ["in"] * 2))
        self.scale = float(rng.choice([0.6, 1.0, 1.6])) if vague else 1.0
        self.size_word = str(rng.choice(SIZE_WORD[self.scale])) if vague else ""
        self.params: list[dict] = []
        self.ops: list[dict] = []
        self.reqs: list[dict] = []
        self.assume: list[str] = []
        self.questions: list[str] = []
        self.t: dict[str, str] = {}

    # --------------------------------------------------------------- parameters
    def _user(self, mm: float) -> float:
        u = self.unit
        if u == "mm":
            step = 1.0 if mm >= 20 else 0.5 if mm >= 3 else 0.1
            return round(round(mm / step) * step, 3)
        if u == "cm":
            return round(mm / 10.0, 1) if mm >= 10 else round(mm / 10.0, 2)
        return round(round(mm / 25.4 / 0.05) * 0.05, 3) if mm >= 6 else round(mm / 25.4, 3)

    def d(self, pid: str, lo: float, hi: float, default: float, what: str = "") -> float:
        """A stated length (precise) or the canonical one (vague). Returns mm."""
        if self.vague:
            mm = round(default * self.scale, 1)
            self.params.append({"id": pid, "kind": "length", "value": mm, "unit": "mm"})
            if what:
                self.assume.append(f"{what} {_fmt(mm)} mm")
            return mm
        stated = self._user(float(self.rng.uniform(lo, hi)))
        if stated <= 0:
            stated = self._user(hi)
        mm = round(stated * UNIT_MM[self.unit], 4)
        self.params.append({"id": pid, "kind": "length", "value": mm, "unit": "mm"})
        self.t[pid] = f"{_fmt(stated)} {self.unit}"
        return mm

    def derived(self, pid: str, mm: float, what: str) -> float:
        """An unstated dimension fixed by a rule of stated ones (recorded as an assumption)."""
        mm = round(mm, 2)
        self.params.append({"id": pid, "kind": "length", "value": mm, "unit": "mm", "status": "derived"})
        self.assume.append(f"{what} {_fmt(mm)} mm")
        return mm

    def n(self, pid: str, lo: int, hi: int, default: int, what: str = "") -> int:
        v = default if self.vague else int(self.rng.integers(lo, hi + 1))
        self.params.append({"id": pid, "kind": "count", "value": v})
        self.t[pid] = str(v)
        if self.vague and what:
            self.assume.append(f"{what} {v}")
        return v

    def ang(self, pid: str, lo: float, hi: float, default: float, what: str = "") -> float:
        v = float(default if self.vague else round(float(self.rng.uniform(lo, hi))))
        self.params.append({"id": pid, "kind": "angle", "value": v, "unit": "deg"})
        self.t[pid] = f"{_fmt(v)} degrees"
        if self.vague and what:
            self.assume.append(f"{what} {_fmt(v)} deg")
        return v

    # --------------------------------------------------------------- ops / requirements
    def op(self, oid: str, op: str, **kw) -> str:
        self.ops.append({"id": oid, "op": op, **kw})
        return oid

    def revolve_up(self, oid: str, start, segments) -> str:
        """Profile in (radius, height) revolved about Y, then turned so the axis is Z."""
        self.op(f"{oid}_p", "profile", start=start, segments=segments)
        self.op(f"{oid}_r", "revolve", profile=f"{oid}_p")
        return self.op(oid, "rotate", input=f"{oid}_r", axis=[1, 0, 0], angle=90)

    def req(self, axis: str, pid: str) -> None:
        if self.vague or pid not in self.t:
            return
        val, unit = self.t[pid].split()
        v = float(val)
        tol = max(0.5 / UNIT_MM[unit], 0.01 * v)
        self.reqs.append({"id": f"r_extent_{axis}", "quantity": f"body.extent_{axis}", "applies_to": self.comp,
                          "comparator": "approx", "target": {"value": v, "unit": unit},
                          "tolerance": {"value": round(tol, 4), "unit": unit}})

    def proposal(self, summary: str, out: str, tags: tuple = ()) -> dict:
        if self.vague:
            self.questions.append("No dimensions were given: confirm the overall size and the key dimensions.")
        comp = {"id": self.comp, "kind": "procedural",
                "geometry": {"units": "mm", "ops": self.ops, "outputs": {"body": out}}}
        if tags:
            comp["tags"] = list(tags)
        return {"schema_version": "0.1.0", "kind": "design_proposal",
                "intent": {"summary": summary, "assumptions": self.assume, "open_questions": self.questions},
                "specification": {"requirements": self.reqs, "parameters": self.params},
                "structure": {"components": [comp]}}

    # --------------------------------------------------------------- text
    def pick(self, options):
        return str(options[int(self.rng.integers(len(options)))])

    def sentence(self, noun: str, details: list[str]) -> str:
        d = [x for x in details if x]
        order = list(self.rng.permutation(len(d))) if len(d) > 2 and self.rng.random() < 0.5 else range(len(d))
        d = [d[i] for i in order]
        head = self.pick(OPEN)
        obj = _a(f"{self.size_word} {noun}".strip()) if not self.vague or self.size_word else _a(noun)
        if head and head[0].isupper() and obj.startswith("a"):
            pass
        body = obj + (" " + ", ".join(d) if d else "")
        txt = (head + body) if head else body[0].upper() + body[1:]
        if self.rng.random() < 0.15 and not self.vague:
            txt += f". All dimensions in {self.unit}" if self.unit != "in" else ". Dimensions in inches"
        return txt + "."


# ===================================================================== hexagon helper
def hexagon(b: Builder, oid: str, flats: str) -> str:
    """Regular hexagon (across flats = expression ``flats``), a vertex on +X, centred at the origin."""
    R, h = f"({flats})/sqrt(3)", f"({flats})/2"
    return b.op(oid, "profile", start=[R, 0], segments=[
        {"kind": "line", "to": [f"{R}/2", h]}, {"kind": "line", "to": [f"-{R}/2", h]},
        {"kind": "line", "to": [f"-{R}", 0]}, {"kind": "line", "to": [f"-{R}/2", f"-{h}"]},
        {"kind": "line", "to": [f"{R}/2", f"-{h}"]}, {"kind": "line", "to": [R, 0]}])


# ===================================================================== families
def plate_holes(b: Builder):
    L = b.d("L", 60, 400, 150, "plate length")
    W = b.d("W", 40, min(L, 300), 100, "plate width")
    T = b.d("T", 3, 20, 8, "plate thickness")
    dmax = min(L, W) / 6
    D = b.d("D", 3, max(3.5, dmax), min(8.5, dmax), "hole diameter")
    nx, ny = b.n("NX", 2, 5, 2), b.n("NY", 2, 4, 2)
    m = b.derived("M", max(1.5 * D, 0.12 * min(L, W)), "hole centre margin from the edges")
    if m * 2 >= min(L, W) - D:
        return None
    b.op("plate", "box", size=["L", "W", "T"], center=True)
    b.op("hole", "cylinder", radius="D/2", height="3*T", center=True)
    b.op("h0", "translate", input="hole", offset=["-L/2+M", "-W/2+M", 0])
    b.op("row", "linear_pattern", input="h0", count=nx, step=[f"(L-2*M)/{nx - 1}", 0, 0])
    b.op("grid", "linear_pattern", input="row", count=ny, step=[0, f"(W-2*M)/{ny - 1}", 0])
    out = b.op("body", "difference", base="plate", tools=["grid"])
    for ax, p in (("x", "L"), ("y", "W"), ("z", "T")):
        b.req(ax, p)
    noun = b.pick(["mounting plate", "base plate", "flat plate", "plate", "adapter plate"])
    t = b.t
    txt = b.sentence(noun, [f"{t.get('L')} by {t.get('W')}" if "L" in t else "",
                            f"{t['T']} thick" if "T" in t else "",
                            f"with a {nx} x {ny} grid of {t['D']} holes" if "D" in t else
                            f"with a {nx} x {ny} hole pattern"])
    return txt, b.proposal(f"plate with {nx}x{ny} through holes", out)


def l_bracket(b: Builder):
    A = b.d("A", 30, 200, 80, "horizontal leg length")
    B_ = b.d("B", 30, 200, 60, "vertical leg height")
    t = b.d("t", 3, 12, 5, "thickness")
    w = b.d("w", 20, 120, 40, "width")
    holes = b.vague or b.rng.random() < 0.7
    b.op("lp", "profile", start=[0, 0], segments=[
        {"kind": "line", "to": ["A", 0]}, {"kind": "line", "to": ["A", "t"]}, {"kind": "line", "to": ["t", "t"]},
        {"kind": "line", "to": ["t", "B"]}, {"kind": "line", "to": [0, "B"]}, {"kind": "line", "to": [0, 0]}])
    out = b.op("bracket", "extrude", profile="lp", height="w")
    if holes:
        hd = b.derived("hd", round(min(w * 0.3, 10.5), 1), "mounting hole diameter")
        if hd < 2:
            return None
        b.op("c1", "cylinder", radius="hd/2", height="4*t", center=True)
        b.op("c1r", "rotate", input="c1", axis=[1, 0, 0], angle=90)
        b.op("c1p", "translate", input="c1r", offset=["0.6*A+0.4*t", "t/2", "w/2"])
        b.op("c2", "cylinder", radius="hd/2", height="4*t", center=True)
        b.op("c2r", "rotate", input="c2", axis=[0, 1, 0], angle=90)
        b.op("c2p", "translate", input="c2r", offset=["t/2", "0.6*B+0.4*t", "w/2"])
        out = b.op("body", "difference", base="bracket", tools=["c1p", "c2p"])
    for ax, p in (("x", "A"), ("y", "B"), ("z", "w")):
        b.req(ax, p)
    T = b.t
    txt = b.sentence(b.pick(["L bracket", "angle bracket", "L-shaped bracket", "corner bracket"]),
                     [f"with legs of {T['A']} and {T['B']}" if "A" in T else "",
                      f"{T['t']} thick" if "t" in T else "", f"{T['w']} wide" if "w" in T else "",
                      "with a mounting hole in each leg" if holes else ""])
    return txt, b.proposal("L bracket" + (" with holes" if holes else ""), out)


def u_channel(b: Builder):
    W = b.d("W", 20, 150, 50, "outer width")
    H = b.d("H", 15, 120, 40, "height")
    t = b.d("t", 2, 10, 4, "wall thickness")
    L = b.d("L", 100, 1000, 300, "length")
    if 2 * t >= W - 2 or t >= H - 2:
        return None
    b.op("up", "profile", start=[0, 0], segments=[
        {"kind": "line", "to": ["W", 0]}, {"kind": "line", "to": ["W", "H"]}, {"kind": "line", "to": ["W-t", "H"]},
        {"kind": "line", "to": ["W-t", "t"]}, {"kind": "line", "to": ["t", "t"]}, {"kind": "line", "to": ["t", "H"]},
        {"kind": "line", "to": [0, "H"]}, {"kind": "line", "to": [0, 0]}])
    out = b.op("body", "extrude", profile="up", height="L")
    for ax, p in (("x", "W"), ("y", "H"), ("z", "L")):
        b.req(ax, p)
    T = b.t
    txt = b.sentence(b.pick(["U channel", "C channel", "U-section profile", "channel section"]),
                     [f"{T['W']} wide and {T['H']} tall" if "W" in T else "",
                      f"with {T['t']} walls" if "t" in T else "", f"{T['L']} long" if "L" in T else ""])
    return txt, b.proposal("U channel", out)


def i_beam(b: Builder):
    bf = b.d("bf", 40, 200, 100, "flange width")
    d = b.d("d", 60, 400, 150, "depth")
    tf = b.d("tf", 4, 20, 8, "flange thickness")
    tw = b.d("tw", 3, 15, 6, "web thickness")
    L = b.d("L", 200, 3000, 1000, "length")
    if 2 * tf >= d - 5 or tw >= bf - 5:
        return None
    s = [["bf/2", 0], ["bf/2", "tf"], ["tw/2", "tf"], ["tw/2", "d-tf"], ["bf/2", "d-tf"], ["bf/2", "d"],
         ["-bf/2", "d"], ["-bf/2", "d-tf"], ["-tw/2", "d-tf"], ["-tw/2", "tf"], ["-bf/2", "tf"], ["-bf/2", 0]]
    b.op("ip", "profile", start=["-bf/2", 0], segments=[{"kind": "line", "to": p} for p in s])
    out = b.op("body", "extrude", profile="ip", height="L")
    for ax, p in (("x", "bf"), ("y", "d"), ("z", "L")):
        b.req(ax, p)
    T = b.t
    txt = b.sentence(b.pick(["I-beam", "H-beam", "wide flange beam", "I section beam"]),
                     [f"{T['d']} deep" if "d" in T else "", f"with {T['bf']} flanges" if "bf" in T else "",
                      f"flange thickness {T['tf']}" if "tf" in T else "", f"web {T['tw']}" if "tw" in T else "",
                      f"{T['L']} long" if "L" in T else ""])
    return txt, b.proposal("I-beam", out)


def flange(b: Builder):
    OD = b.d("OD", 60, 300, 150, "outer diameter")
    T = b.d("T", 6, 30, 14, "plate thickness")
    bore = b.d("bore", 10, OD * 0.4, OD * 0.3, "bore diameter")
    nb = b.n("nb", 3, 12, 6, "bolt count")
    hub = b.vague or b.rng.random() < 0.5
    hd = b.derived("hd", round(max(4.0, min(OD * 0.08, 22)), 1), "bolt hole diameter")
    hubd = b.derived("hubd", round(bore * 1.6, 1), "hub diameter") if hub else bore
    pcd = b.derived("PCD", round((OD + max(hubd, bore)) / 2, 1), "bolt circle diameter")
    if pcd / 2 + hd / 2 > OD / 2 - 2 or pcd / 2 - hd / 2 < max(hubd, bore) / 2 + 1:
        return None
    b.op("disk", "cylinder", radius="OD/2", height="T")
    base = "disk"
    if hub:
        hh = b.derived("hubh", round(T * 0.8, 1), "hub projection")
        b.op("hub", "cylinder", radius="hubd/2", height="T+hubh")
        base = b.op("blank", "union", inputs=["disk", "hub"])
        del hh
    b.op("bore_c", "cylinder", radius="bore/2", height="6*T", center=True)
    b.op("bh", "cylinder", radius="hd/2", height="6*T", center=True)
    b.op("bh0", "translate", input="bh", offset=["PCD/2", 0, 0])
    b.op("bolts", "polar_pattern", input="bh0", count=nb)
    out = b.op("body", "difference", base=base, tools=["bore_c", "bolts"])
    b.req("x", "OD")
    if not hub:
        b.req("z", "T")
    t = b.t
    txt = b.sentence(b.pick(["pipe flange", "flange", "bolted flange", "round flange plate"]),
                     [f"{t['OD']} outer diameter" if "OD" in t else "", f"{t['T']} thick" if "T" in t else "",
                      f"with a {t['bore']} bore" if "bore" in t else "",
                      f"and {nb} bolt holes" if not b.vague else "", "with a raised hub" if hub else ""])
    return txt, b.proposal(f"flange, {nb} bolt holes" + (", hub" if hub else ""), out)


def washer(b: Builder):
    OD = b.d("OD", 8, 120, 24, "outer diameter")
    ID = b.d("ID", 3, OD * 0.7, OD * 0.45, "inner diameter")
    T = b.d("T", 0.5, 6, 2, "thickness")
    if ID >= OD - 1:
        return None
    b.op("o", "cylinder", radius="OD/2", height="T")
    b.op("i", "cylinder", radius="ID/2", height="4*T", center=True)
    out = b.op("body", "difference", base="o", tools=["i"])
    b.req("x", "OD")
    b.req("z", "T")
    t = b.t
    txt = b.sentence(b.pick(["washer", "flat washer", "shim ring", "ring washer"]),
                     [f"{t['OD']} OD" if "OD" in t else "", f"{t['ID']} ID" if "ID" in t else "",
                      f"{t['T']} thick" if "T" in t else ""])
    return txt, b.proposal("flat washer", out)


def bushing(b: Builder):
    OD = b.d("OD", 8, 120, 30, "outer diameter")
    ID = b.d("ID", 3, OD * 0.8, OD * 0.55, "inner diameter")
    L = b.d("L", 10, 200, 40, "length")
    if ID >= OD - 1:
        return None
    b.op("o", "cylinder", radius="OD/2", height="L")
    b.op("i", "cylinder", radius="ID/2", height="3*L", center=True)
    out = b.op("body", "difference", base="o", tools=["i"])
    b.req("x", "OD")
    b.req("z", "L")
    t = b.t
    txt = b.sentence(b.pick(["bushing", "spacer tube", "sleeve", "tubular spacer", "pipe section"]),
                     [f"{t['OD']} outside diameter" if "OD" in t else "",
                      f"{t['ID']} inside diameter" if "ID" in t else "", f"{t['L']} long" if "L" in t else ""])
    return txt, b.proposal("bushing", out)


def stepped_shaft(b: Builder):
    k = b.n("steps", 2, 4, 3)
    ds, ls = [], []
    for i in range(k):
        ds.append(b.d(f"D{i + 1}", 8, 80, [30, 40, 25, 20][i], f"diameter of section {i + 1}"))
        ls.append(b.d(f"L{i + 1}", 10, 150, [60, 40, 50, 30][i], f"length of section {i + 1}"))
    segs, y = [{"kind": "line", "to": ["D1/2", 0]}], "0"
    for i in range(k):
        y = f"{y}+L{i + 1}" if i else f"L{i + 1}"
        segs.append({"kind": "line", "to": [f"D{i + 1}/2", y]})
        if i + 1 < k:
            segs.append({"kind": "line", "to": [f"D{i + 2}/2", y]})
    segs += [{"kind": "line", "to": [0, y]}, {"kind": "line", "to": [0, 0]}]
    out = b.revolve_up("body", [0, 0], segs)
    if not b.vague:
        tot = sum(ls)
        b.params.append({"id": "Ltot", "kind": "length", "value": round(tot, 4), "unit": "mm", "status": "derived"})
        b.reqs.append({"id": "r_extent_z", "quantity": "body.extent_z", "applies_to": b.comp, "comparator": "approx",
                       "target": {"value": round(tot / UNIT_MM[b.unit], 4), "unit": b.unit},
                       "tolerance": {"value": round(max(0.5 / UNIT_MM[b.unit], 0.01 * tot / UNIT_MM[b.unit]), 4),
                                     "unit": b.unit}})
    t = b.t
    secs = [f"{t[f'D{i + 1}']} diameter x {t[f'L{i + 1}']}" for i in range(k)] if not b.vague else []
    txt = b.sentence(b.pick(["stepped shaft", "shaft", "turned shaft", "spindle"]),
                     [f"with {k} sections" + (": " + "; ".join(secs) if secs else "")])
    return txt, b.proposal(f"stepped shaft, {k} sections", out)


def hex_nut(b: Builder):
    s = b.d("s", 5, 60, 17, "width across flats")
    m = b.d("m", 3, 40, 8, "thickness")
    d = b.derived("d", round(s * 0.6, 2), "bore diameter")
    hexagon(b, "hx", "s")
    b.op("prism", "extrude", profile="hx", height="m")
    b.op("bore_c", "cylinder", radius="d/2", height="4*m", center=True)
    out = b.op("body", "difference", base="prism", tools=["bore_c"])
    b.req("y", "s")
    b.req("z", "m")
    t = b.t
    txt = b.sentence(b.pick(["hex nut", "hexagon nut", "nut", "hex spacer nut"]),
                     [f"{t['s']} across flats" if "s" in t else "", f"{t['m']} thick" if "m" in t else ""])
    return txt, b.proposal("hex nut", out)


def hex_bolt(b: Builder):
    d = b.d("d", 4, 30, 10, "shank diameter")
    L = b.d("L", 10, 200, 50, "shank length")
    s = b.derived("s", round(1.6 * d, 2), "head across flats")
    b.derived("k", round(0.65 * d, 2), "head height")
    b.op("shank", "cylinder", radius="d/2", height="L")
    hexagon(b, "hx", "s")
    b.op("head0", "extrude", profile="hx", height="k")
    b.op("head", "translate", input="head0", offset=[0, 0, "L"])
    out = b.op("body", "union", inputs=["shank", "head"])
    t = b.t
    txt = b.sentence(b.pick(["hex bolt", "hex head bolt", "bolt", "hex cap screw (unthreaded)"]),
                     [f"{t['d']} diameter" if "d" in t else "", f"{t['L']} long under the head" if "L" in t else ""])
    return txt, b.proposal("hex bolt (no thread)", out)


def pulley(b: Builder):
    OD = b.d("OD", 30, 300, 100, "outer diameter")
    w = b.d("w", 8, 60, 20, "width")
    bore = b.d("bore", 5, OD * 0.35, OD * 0.2, "bore diameter")
    g = b.derived("g", round(min(w * 0.4, OD * 0.15), 2), "groove depth")
    if bore / 2 >= OD / 2 - g - 2:
        return None
    out = b.revolve_up("body", ["bore/2", 0], [
        {"kind": "line", "to": ["OD/2", 0]}, {"kind": "line", "to": ["OD/2", "0.15*w"]},
        {"kind": "line", "to": ["OD/2-g", "0.5*w"]}, {"kind": "line", "to": ["OD/2", "0.85*w"]},
        {"kind": "line", "to": ["OD/2", "w"]}, {"kind": "line", "to": ["bore/2", "w"]},
        {"kind": "line", "to": ["bore/2", 0]}])
    b.req("x", "OD")
    b.req("z", "w")
    t = b.t
    txt = b.sentence(b.pick(["V-belt pulley", "pulley", "grooved pulley", "belt sheave"]),
                     [f"{t['OD']} diameter" if "OD" in t else "", f"{t['w']} wide" if "w" in t else "",
                      f"{t['bore']} bore" if "bore" in t else ""])
    return txt, b.proposal("V-groove pulley", out)


def spoked_wheel(b: Builder):
    OD = b.d("OD", 80, 600, 250, "outer diameter")
    rw = b.d("rw", 10, 80, 30, "rim width")
    n = b.n("n", 3, 12, 6, "spoke count")
    rt = b.derived("rt", round(OD * 0.06, 2), "rim thickness")
    hubd = b.derived("hubd", round(OD * 0.2, 2), "hub diameter")
    sw = b.derived("sw", round(max(OD * 0.03, 3), 2), "spoke width")
    b.op("rim_o", "cylinder", radius="OD/2", height="rw")
    b.op("rim_i", "cylinder", radius="OD/2-rt", height="3*rw", center=True)
    b.op("rim", "difference", base="rim_o", tools=["rim_i"])
    b.op("hub", "cylinder", radius="hubd/2", height="rw")
    b.op("spoke", "box", size=["OD/2-rt/2", "sw", "0.6*rw"], center=True)
    b.op("spoke0", "translate", input="spoke", offset=["(OD/2-rt/2)/2", 0, "rw/2"])
    b.op("spokes", "polar_pattern", input="spoke0", count=n)
    b.op("whole", "union", inputs=["rim", "hub", "spokes"])
    b.op("bore_c", "cylinder", radius="hubd/4", height="3*rw", center=True)
    out = b.op("body", "difference", base="whole", tools=["bore_c"])
    b.req("x", "OD")
    b.req("z", "rw")
    t = b.t
    txt = b.sentence(b.pick(["spoked wheel", "wheel", "handwheel", "flywheel with spokes"]),
                     [f"{t['OD']} across" if "OD" in t else "", f"{t['rw']} wide rim" if "rw" in t else "",
                      f"{n} spokes" if not b.vague else ""])
    return txt, b.proposal(f"wheel, {n} spokes", out)


def pillow_block(b: Builder):
    D = b.d("D", 10, 60, 25, "bore diameter")
    Hc = b.d("Hc", 20, 120, 40, "shaft centre height")
    Lb = b.derived("Lb", round(D * 4.5, 1), "base length")
    Wb = b.derived("Wb", round(D * 1.4, 1), "block width")
    Tb = b.derived("Tb", round(max(D * 0.5, 6), 1), "base thickness")
    Db = b.derived("Db", round(D * 2.0, 1), "housing diameter")
    if Hc < Tb + 2 or Hc - Db / 2 < 0:
        return None
    b.op("base", "box", size=["Lb", "Wb", "Tb"], center=True)
    b.op("base_p", "translate", input="base", offset=[0, 0, "Tb/2"])
    b.op("ped", "box", size=["Db", "Wb", "Hc"], center=True)
    b.op("ped_p", "translate", input="ped", offset=[0, 0, "Hc/2"])
    b.op("boss", "cylinder", radius="Db/2", height="Wb", center=True)
    b.op("boss_r", "rotate", input="boss", axis=[1, 0, 0], angle=90)
    b.op("boss_p", "translate", input="boss_r", offset=[0, 0, "Hc"])
    b.op("solid", "union", inputs=["base_p", "ped_p", "boss_p"])
    b.op("bore_c", "cylinder", radius="D/2", height="3*Wb", center=True)
    b.op("bore_r", "rotate", input="bore_c", axis=[1, 0, 0], angle=90)
    b.op("bore_p", "translate", input="bore_r", offset=[0, 0, "Hc"])
    b.op("bolt", "cylinder", radius="Tb*0.35", height="4*Tb", center=True)
    b.op("bolt_p", "translate", input="bolt", offset=["-(Lb/2+Db/2)/2", 0, 0])
    b.op("bolts", "linear_pattern", input="bolt_p", count=2, step=["(Lb/2+Db/2)", 0, 0])
    out = b.op("body", "difference", base="solid", tools=["bore_p", "bolts"])
    t = b.t
    txt = b.sentence(b.pick(["pillow block", "bearing housing", "shaft support block", "plummer block"]),
                     [f"for a {t['D']} shaft" if "D" in t else "", f"centre height {t['Hc']}" if "Hc" in t else "",
                      "with two bolt holes in the base"])
    return txt, b.proposal("pillow block", out)


def sprocket(b: Builder):
    D = b.d("D", 40, 300, 120, "root diameter")
    T = b.d("T", 3, 20, 8, "thickness")
    n = b.n("n", 8, 48, 24, "tooth count")
    tl = b.derived("tl", round(D * 0.07, 2), "tooth height")
    tw = b.derived("tw", round(math.pi * D / n * 0.45, 2), "tooth width")
    b.op("disk", "cylinder", radius="D/2", height="T")
    b.op("tooth", "box", size=["tl+1[mm]", "tw", "T"], center=True)
    b.op("tooth0", "translate", input="tooth", offset=["D/2+tl/2-0.5[mm]", 0, "T/2"])
    b.op("teeth", "polar_pattern", input="tooth0", count=n)
    b.op("blank", "union", inputs=["disk", "teeth"])
    b.op("bore_c", "cylinder", radius="D/8", height="4*T", center=True)
    out = b.op("body", "difference", base="blank", tools=["bore_c"])
    b.req("z", "T")
    t = b.t
    txt = b.sentence(b.pick(["toothed disc", "sprocket-like toothed wheel", "ratchet disc", "toothed wheel"]),
                     [f"{n} teeth" if not b.vague else "", f"{t['D']} root diameter" if "D" in t else "",
                      f"{t['T']} thick" if "T" in t else ""])
    return txt, b.proposal(f"toothed disc, {n} teeth", out)


def heat_sink(b: Builder):
    Lx = b.d("Lx", 30, 200, 80, "base length")
    Wy = b.d("Wy", 20, 200, 60, "base width")
    Hf = b.d("Hf", 8, 60, 25, "fin height")
    n = b.n("n", 4, 20, 9, "fin count")
    tb = b.derived("tb", round(max(Lx * 0.04, 2), 2), "base thickness")
    ft = b.derived("ft", round(min(Lx / n * 0.35, 3), 2), "fin thickness")
    b.op("base", "box", size=["Lx", "Wy", "tb"], center=True)
    b.op("base_p", "translate", input="base", offset=[0, 0, "tb/2"])
    b.op("fin", "box", size=["ft", "Wy", "Hf"], center=True)
    b.op("fin0", "translate", input="fin", offset=["-Lx/2+ft/2", 0, "tb+Hf/2-0.01[mm]"])
    b.op("fins", "linear_pattern", input="fin0", count=n, step=[f"(Lx-ft)/{n - 1}", 0, 0])
    out = b.op("body", "union", inputs=["base_p", "fins"])
    b.req("x", "Lx")
    b.req("y", "Wy")
    t = b.t
    txt = b.sentence(b.pick(["heat sink", "finned heatsink", "cooling fin block", "extruded heat sink"]),
                     [f"{t['Lx']} x {t['Wy']} base" if "Lx" in t else "",
                      f"{n} fins {t['Hf']} tall" if "Hf" in t else f"{n} fins" if not b.vague else ""])
    return txt, b.proposal(f"heat sink, {n} fins", out)


def enclosure(b: Builder):
    X = b.d("X", 40, 300, 120, "outer length")
    Y = b.d("Y", 30, 250, 80, "outer width")
    Z = b.d("Z", 20, 150, 50, "outer height")
    t = b.d("t", 1.5, 6, 2.5, "wall thickness")
    if 2 * t >= min(X, Y) - 4 or t >= Z - 4:
        return None
    b.op("outer", "box", size=["X", "Y", "Z"], center=True)
    b.op("outer_p", "translate", input="outer", offset=[0, 0, "Z/2"])
    b.op("inner", "box", size=["X-2*t", "Y-2*t", "Z"], center=True)
    b.op("inner_p", "translate", input="inner", offset=[0, 0, "Z/2+t"])
    out = b.op("body", "difference", base="outer_p", tools=["inner_p"])
    for ax, p in (("x", "X"), ("y", "Y"), ("z", "Z")):
        b.req(ax, p)
    T = b.t
    txt = b.sentence(b.pick(["open-top enclosure", "project box without lid", "electronics enclosure base",
                             "rectangular tray", "open box"]),
                     [f"{T['X']} x {T['Y']} x {T['Z']}" if "X" in T else "",
                      f"with {T['t']} walls" if "t" in T else ""])
    return txt, b.proposal("open enclosure", out)


def honeycomb_panel(b: Builder):
    X = b.d("X", 80, 300, 150, "panel length")
    Y = b.d("Y", 60, 250, 100, "panel width")
    T = b.d("T", 3, 20, 6, "panel thickness")
    c = b.d("c", 6, 30, 12, "cell size across flats")
    wall = b.derived("wall", round(max(c * 0.2, 1.2), 2), "cell wall")
    p = c + wall
    nx, ny = int((X - 2 * p) // p), int((Y - 2 * p) // (p * math.sqrt(3)))
    if nx < 2 or ny < 1:
        return None
    b.op("plate", "box", size=["X", "Y", "T"], center=True)
    hexagon(b, "hx", "c")
    b.op("cell", "extrude", profile="hx", height="3*T")
    b.op("cell0", "translate", input="cell", offset=[f"-{nx - 1}*(c+wall)/2", f"-{ny}*(c+wall)*sqrt(3)/2", "-1.5*T"])
    b.op("rowa", "linear_pattern", input="cell0", count=nx, step=["c+wall", 0, 0])
    b.op("rowb", "translate", input="rowa", offset=["(c+wall)/2", "(c+wall)*sqrt(3)/2", 0])
    b.op("pair", "union", inputs=["rowa", "rowb"])
    b.op("cells", "linear_pattern", input="pair", count=ny, step=[0, "(c+wall)*sqrt(3)", 0])
    out = b.op("body", "difference", base="plate", tools=["cells"])
    for ax, pp in (("x", "X"), ("y", "Y"), ("z", "T")):
        b.req(ax, pp)
    t = b.t
    txt = b.sentence(b.pick(["honeycomb panel", "hex-perforated plate", "lightweight honeycomb plate",
                             "plate with hexagonal cutouts"]),
                     [f"{t['X']} by {t['Y']}" if "X" in t else "", f"{t['T']} thick" if "T" in t else "",
                      f"{t['c']} hex cells" if "c" in t else ""])
    return txt, b.proposal("honeycomb panel", out)


def pipe_tee(b: Builder):
    OD = b.d("OD", 15, 200, 60, "outer diameter")
    ID = b.d("ID", 8, OD * 0.85, OD * 0.75, "inner diameter")
    Lm = b.d("Lm", OD * 2, OD * 5, OD * 3, "run length")
    Lb = b.d("Lb", OD * 1, OD * 3, OD * 1.5, "branch length from the run axis")
    if ID >= OD - 1:
        return None
    b.op("mo", "cylinder", radius="OD/2", height="Lm", center=True)
    b.op("mo_r", "rotate", input="mo", axis=[0, 1, 0], angle=90)
    b.op("bo", "cylinder", radius="OD/2", height="Lb")
    b.op("outer", "union", inputs=["mo_r", "bo"])
    b.op("mi", "cylinder", radius="ID/2", height="Lm+2[mm]", center=True)
    b.op("mi_r", "rotate", input="mi", axis=[0, 1, 0], angle=90)
    b.op("bi", "cylinder", radius="ID/2", height="Lb+1[mm]")
    out = b.op("body", "difference", base="outer", tools=["mi_r", "bi"])
    b.req("x", "Lm")
    t = b.t
    txt = b.sentence(b.pick(["pipe tee", "T-fitting", "tee junction", "equal tee"]),
                     [f"{t['OD']} OD, {t['ID']} ID" if "OD" in t else "", f"run length {t['Lm']}" if "Lm" in t else "",
                      f"branch {t['Lb']} from the centreline" if "Lb" in t else ""])
    return txt, b.proposal("pipe tee", out)


def pipe_elbow(b: Builder):
    OD = b.d("OD", 15, 150, 50, "outer diameter")
    ID = b.d("ID", 8, OD * 0.85, OD * 0.75, "inner diameter")
    R = b.d("R", OD * 1.0, OD * 3, OD * 1.5, "bend radius")
    if ID >= OD - 1 or R <= OD / 2 + 1:
        return None
    b.op("to", "torus", major_radius="R", minor_radius="OD/2")
    b.op("ti", "torus", major_radius="R", minor_radius="ID/2")
    b.op("ring", "difference", base="to", tools=["ti"])
    b.op("q", "box", size=["R+OD", "R+OD", "2*OD"], center=False)
    b.op("q_p", "translate", input="q", offset=[0, 0, "-OD"])
    out = b.op("body", "intersection", inputs=["ring", "q_p"])
    t = b.t
    txt = b.sentence(b.pick(["90 degree pipe elbow", "pipe bend", "quarter bend", "elbow fitting"]),
                     [f"{t['OD']} OD" if "OD" in t else "", f"{t['ID']} ID" if "ID" in t else "",
                      f"bend radius {t['R']}" if "R" in t else ""])
    return txt, b.proposal("90 deg pipe elbow", out)


def pressure_vessel(b: Builder):
    R = b.d("R", 50, 600, 200, "outer radius")
    Ls = b.d("Ls", 50, 1500, 500, "cylindrical length")
    t = b.d("t", 3, 30, 8, "wall thickness")
    rn = b.derived("rn", round(max(R * 0.15, t + 4), 2), "nozzle outer radius")
    b.derived("hn", round(R * 0.3, 2), "nozzle height above the head")
    if Ls < 5 or t >= R * 0.3 or rn - t < 2:
        return None
    k = "0.70710678"

    def capsule(oid, r):
        return b.revolve_up(oid, [0, f"-Ls/2-{r}"], [
            {"kind": "arc", "through": [f"{r}*{k}", f"-Ls/2-{r}*{k}"], "to": [r, "-Ls/2"]},
            {"kind": "line", "to": [r, "Ls/2"]},
            {"kind": "arc", "through": [f"{r}*{k}", f"Ls/2+{r}*{k}"], "to": [0, f"Ls/2+{r}"]},
            {"kind": "line", "to": [0, f"-Ls/2-{r}"]}])
    capsule("outer", "R")
    capsule("inner", "(R-t)")
    b.op("shell", "difference", base="outer", tools=["inner"])
    b.op("noz", "cylinder", radius="rn", height="R+hn")
    b.op("noz_p", "translate", input="noz", offset=[0, 0, "Ls/2"])
    b.op("solid", "union", inputs=["shell", "noz_p"])
    b.op("nb", "cylinder", radius="rn-t", height="R+hn+t")
    b.op("nb_p", "translate", input="nb", offset=[0, 0, "Ls/2"])
    out = b.op("body", "difference", base="solid", tools=["nb_p"])
    T = b.t
    if "R" in T:
        val, unit = T["R"].split()
        dia = f"{_fmt(2 * float(val))} {unit}"
        b.reqs.append({"id": "r_extent_x", "quantity": "body.extent_x", "applies_to": b.comp, "comparator": "approx",
                       "target": {"value": 2 * float(val), "unit": unit},
                       "tolerance": {"value": round(max(0.5 / UNIT_MM[unit], 0.02 * float(val)), 4), "unit": unit}})
    txt = b.sentence(b.pick(["pressure vessel", "cylindrical tank with hemispherical heads", "air receiver tank",
                             "capsule-shaped pressure tank"]),
                     [f"{dia} outside diameter" if "R" in T else "",
                      f"{T['Ls']} of straight shell" if "Ls" in T else "",
                      f"{T['t']} wall" if "t" in T else "", "with a nozzle on top"])
    return txt, b.proposal("pressure vessel with hemispherical heads and top nozzle", out)


def funnel(b: Builder):
    Rt = b.d("Rt", 30, 200, 70, "top radius")
    Hc = b.d("Hc", 30, 250, 80, "cone height")
    rs = b.derived("rs", round(max(Rt * 0.18, 6), 2), "spout outer radius")
    b.derived("Lsp", round(Hc * 0.6, 2), "spout length")
    b.derived("t", round(max(Rt * 0.03, 1.5), 2), "wall thickness")
    out = b.revolve_up("body", ["rs", 0], [
        {"kind": "line", "to": ["rs", "Lsp"]}, {"kind": "line", "to": ["Rt", "Lsp+Hc"]},
        {"kind": "line", "to": ["Rt-t", "Lsp+Hc"]}, {"kind": "line", "to": ["rs-t", "Lsp+t"]},
        {"kind": "line", "to": ["rs-t", 0]}, {"kind": "line", "to": ["rs", 0]}])
    T = b.t
    if "Rt" in T:
        val, unit = T["Rt"].split()
        dia = f"{_fmt(2 * float(val))} {unit}"
        b.reqs.append({"id": "r_extent_x", "quantity": "body.extent_x", "applies_to": b.comp, "comparator": "approx",
                       "target": {"value": 2 * float(val), "unit": unit},
                       "tolerance": {"value": round(max(0.5 / UNIT_MM[unit], 0.02 * float(val)), 4), "unit": unit}})
    txt = b.sentence(b.pick(["funnel", "conical funnel", "pouring funnel", "hopper funnel"]),
                     [f"{dia} across the top" if "Rt" in T else "", f"cone {T['Hc']} tall" if "Hc" in T else "",
                      "with a short spout"])
    return txt, b.proposal("funnel", out)


def rocket_nozzle(b: Builder):
    rt = b.d("rt", 5, 60, 20, "throat radius")
    eps = round(float(b.rng.uniform(3, 25)), 1) if not b.vague else 8.0
    b.params.append({"id": "eps", "kind": "real", "value": eps})
    b.t["eps"] = _fmt(eps)
    b.derived("re", round(rt * math.sqrt(eps), 2), "exit radius (from the area ratio)")
    b.derived("ri", round(rt * 1.8, 2), "chamber-side inlet radius")
    b.derived("Lc", round(rt * 1.5, 2), "converging length")
    b.derived("Ld", round(rt * (math.sqrt(eps) - 1) / math.tan(math.radians(15)) * 0.8, 2), "diverging length")
    b.derived("t", round(max(rt * 0.12, 1.5), 2), "wall thickness")
    out = b.revolve_up("body", ["ri", 0], [
        {"kind": "spline", "points": [["(ri+rt)/2+0.1*rt", "0.6*Lc"], ["rt", "Lc"]]},
        {"kind": "spline", "points": [["rt+0.45*(re-rt)", "Lc+0.35*Ld"], ["re", "Lc+Ld"]]},
        {"kind": "line", "to": ["re+t", "Lc+Ld"]},
        {"kind": "spline", "points": [["rt+0.45*(re-rt)+t", "Lc+0.35*Ld"], ["rt+t", "Lc"]]},
        {"kind": "spline", "points": [["(ri+rt)/2+0.1*rt+t", "0.6*Lc"], ["ri+t", 0]]},
        {"kind": "line", "to": ["ri", 0]}])
    T = b.t
    txt = b.sentence(b.pick(["rocket engine nozzle", "convergent-divergent nozzle", "de Laval nozzle", "bell nozzle"]),
                     [f"throat radius {T['rt']}" if "rt" in T else "",
                      f"expansion (area) ratio {T['eps']}" if not b.vague else ""])
    return txt, b.proposal(f"C-D nozzle, area ratio {eps:g}", out)


def rocket_body(b: Builder):
    Rb = b.d("Rb", 10, 150, 40, "body radius")
    Lb = b.d("Lb", 80, 1500, 400, "body tube length")
    nf = b.n("nf", 3, 4, 4, "fin count")
    Ln = b.derived("Ln", round(Rb * 4, 2), "nose cone length")
    b.derived("fs", round(Rb * 1.2, 2), "fin span")
    b.derived("fr", round(min(Rb * 2.5, Lb * 0.4), 2), "fin root chord")
    b.derived("ft", round(max(Rb * 0.08, 1.5), 2), "fin thickness")
    b.derived("He", round(Rb * 0.8, 2), "engine bell length")
    del Ln
    b.revolve_up("hull", [0, 0], [
        {"kind": "line", "to": ["Rb", 0]}, {"kind": "line", "to": ["Rb", "Lb"]},
        {"kind": "spline", "points": [["Rb*0.78", "Lb+Ln*0.5"], ["Rb*0.32", "Lb+Ln*0.88"], [0, "Lb+Ln"]]},
        {"kind": "line", "to": [0, 0]}])
    b.op("fin", "box", size=["fs+1[mm]", "ft", "fr"], center=True)
    b.op("fin0", "translate", input="fin", offset=["Rb+fs/2-0.5[mm]", 0, "fr/2"])
    b.op("fins", "polar_pattern", input="fin0", count=nf)
    b.op("bell", "cone", radius1="Rb*0.6", radius2="Rb*0.35", height="He")
    b.op("bell_p", "translate", input="bell", offset=[0, 0, "-He+0.01[mm]"])
    out = b.op("body", "union", inputs=["hull", "fins", "bell_p"])
    T = b.t
    txt = b.sentence(b.pick(["model rocket", "rocket", "sounding rocket airframe", "rocket body with fins"]),
                     [f"{T['Rb']} body radius" if "Rb" in T else "", f"{T['Lb']} body tube" if "Lb" in T else "",
                      f"{nf} fins" if not b.vague else "", "an ogive nose cone and an engine bell"])
    return txt, b.proposal(f"rocket airframe, {nf} fins", out)


def propeller(b: Builder):
    D = b.d("D", 60, 600, 200, "diameter")
    n = b.n("n", 2, 6, 3, "blade count")
    pitch = b.ang("pitch", 10, 40, 20, "blade pitch")
    rh = b.derived("rh", round(D * 0.08, 2), "hub radius")
    b.derived("hh", round(D * 0.08, 2), "hub length")
    b.derived("ch", round(D * 0.12, 2), "blade chord")
    b.derived("bt", round(max(D * 0.012, 1.2), 2), "blade thickness")
    b.op("hub", "cylinder", radius="rh", height="hh", center=True)
    b.op("blade", "box", size=["D/2-rh*0.8", "ch", "bt"], center=True)
    b.op("blade_t", "rotate", input="blade", axis=[1, 0, 0], angle="pitch")
    b.op("blade0", "translate", input="blade_t", offset=["(D/2-rh*0.8)/2+rh*0.8", 0, 0])
    b.op("blades", "polar_pattern", input="blade0", count=n)
    b.op("whole", "union", inputs=["hub", "blades"])
    b.op("shaft", "cylinder", radius="rh*0.35", height="3*hh", center=True)
    out = b.op("body", "difference", base="whole", tools=["shaft"])
    if n % 2 == 0:
        b.req("x", "D")
    T = b.t
    txt = b.sentence(b.pick(["propeller", "fan rotor", "boat propeller", "drone propeller"]),
                     [f"{n} blades" if not b.vague else "", f"{T['D']} diameter" if "D" in T else "",
                      f"{T['pitch']} blade pitch" if "pitch" in T else ""])
    return txt, b.proposal(f"{n}-blade propeller", out)


def mug(b: Builder):
    R = b.d("R", 30, 60, 42, "outer radius")
    H = b.d("H", 60, 150, 95, "height")
    t = b.derived("t", round(max(R * 0.07, 2.5), 2), "wall thickness")
    b.derived("a", round(H * 0.28, 2), "handle loop radius")
    b.derived("c", round(max(t * 1.6, 4), 2), "handle thickness")
    b.revolve_up("cup", [0, 0], [
        {"kind": "line", "to": ["R", 0]}, {"kind": "line", "to": ["R", "H"]}, {"kind": "line", "to": ["R-t", "H"]},
        {"kind": "line", "to": ["R-t", "t"]}, {"kind": "line", "to": [0, "t"]}, {"kind": "line", "to": [0, 0]}])
    b.op("loop", "torus", major_radius="a", minor_radius="c")
    b.op("loop_r", "rotate", input="loop", axis=[1, 0, 0], angle=90)
    b.op("loop_p", "translate", input="loop_r", offset=["R", 0, "H/2"])
    b.op("keep", "box", size=["2*a+2*c", "4*c", "2*a+2*c+2[mm]"], center=True)
    b.op("keep_p", "translate", input="keep", offset=["R-t/2+a+c", 0, "H/2"])
    b.op("handle", "intersection", inputs=["loop_p", "keep_p"])
    out = b.op("body", "union", inputs=["cup", "handle"])
    b.req("z", "H")
    T = b.t
    txt = b.sentence(b.pick(["coffee mug", "mug", "cup with a handle", "tea mug"]),
                     [f"{T['H']} tall" if "H" in T else "", f"outer radius {T['R']}" if "R" in T else ""])
    return txt, b.proposal("mug with handle", out)


def bottle(b: Builder):
    Rb = b.d("Rb", 20, 60, 35, "body radius")
    H = b.d("H", 100, 350, 220, "height")
    rn = b.derived("rn", round(max(Rb * 0.35, 8), 2), "neck outer radius")
    b.derived("h1", round(H * 0.6, 2), "shoulder start height")
    b.derived("h2", round(H * 0.82, 2), "neck start height")
    b.derived("t", round(max(Rb * 0.05, 1.5), 2), "wall thickness")
    if rn >= Rb - 3:
        return None
    out = b.revolve_up("body", [0, 0], [
        {"kind": "line", "to": ["Rb", 0]}, {"kind": "line", "to": ["Rb", "h1"]},
        {"kind": "spline", "points": [["(Rb+rn)/2+0.15*Rb", "(h1+h2)/2"], ["rn", "h2"]]},
        {"kind": "line", "to": ["rn", "H"]}, {"kind": "line", "to": ["rn-t", "H"]}, {"kind": "line", "to": ["rn-t", "h2"]},
        {"kind": "spline", "points": [["(Rb+rn)/2+0.15*Rb-t", "(h1+h2)/2"], ["Rb-t", "h1"]]},
        {"kind": "line", "to": ["Rb-t", "t"]}, {"kind": "line", "to": [0, "t"]}, {"kind": "line", "to": [0, 0]}])
    b.req("z", "H")
    T = b.t
    txt = b.sentence(b.pick(["bottle", "water bottle", "glass bottle", "round bottle"]),
                     [f"{T['H']} tall" if "H" in T else "", f"body radius {T['Rb']}" if "Rb" in T else "",
                      "with a narrow neck"])
    return txt, b.proposal("bottle", out)


def vase(b: Builder):
    H = b.d("H", 80, 400, 200, "height")
    Rm = b.d("Rm", 25, 150, 60, "maximum radius")
    b.derived("Rb0", round(Rm * 0.55, 2), "foot radius")
    b.derived("Rt", round(Rm * 0.6, 2), "mouth radius")
    b.derived("t", round(max(Rm * 0.05, 2), 2), "wall thickness")
    out = b.revolve_up("body", [0, 0], [
        {"kind": "line", "to": ["Rb0", 0]},
        {"kind": "spline", "points": [["Rm", "0.38*H"], ["0.8*Rm", "0.75*H"], ["Rt", "H"]]},
        {"kind": "line", "to": ["Rt-t", "H"]},
        {"kind": "spline", "points": [["0.8*Rm-t", "0.75*H"], ["Rm-t", "0.38*H"], ["Rb0-t", "t"]]},
        {"kind": "line", "to": [0, "t"]}, {"kind": "line", "to": [0, 0]}])
    b.req("z", "H")
    T = b.t
    txt = b.sentence(b.pick(["vase", "flower vase", "bulbous vase", "decorative vase"]),
                     [f"{T['H']} tall" if "H" in T else "", f"bulging to a {T['Rm']} radius" if "Rm" in T else ""])
    return txt, b.proposal("vase", out)


def table(b: Builder):
    L = b.d("L", 400, 2400, 1200, "top length")
    W = b.d("W", 300, 1200, 700, "top width")
    H = b.d("H", 300, 1100, 740, "height")
    b.derived("tt", round(max(min(L, W) * 0.035, 15), 1), "top thickness")
    lw = b.derived("lw", round(max(min(L, W) * 0.06, 30), 1), "leg size")
    b.derived("ins", round(lw, 1), "leg inset")
    b.op("top", "box", size=["L", "W", "tt"], center=True)
    b.op("top_p", "translate", input="top", offset=[0, 0, "H-tt/2"])
    b.op("leg", "box", size=["lw", "lw", "H-tt+0.01[mm]"], center=True)
    b.op("leg0", "translate", input="leg", offset=["-L/2+ins", "-W/2+ins", "(H-tt)/2"])
    b.op("legs_x", "linear_pattern", input="leg0", count=2, step=["L-2*ins", 0, 0])
    b.op("legs", "linear_pattern", input="legs_x", count=2, step=[0, "W-2*ins", 0])
    out = b.op("body", "union", inputs=["top_p", "legs"])
    for ax, p in (("x", "L"), ("y", "W"), ("z", "H")):
        b.req(ax, p)
    T = b.t
    txt = b.sentence(b.pick(["table", "dining table", "desk", "work table"]),
                     [f"{T['L']} x {T['W']}" if "L" in T else "", f"{T['H']} high" if "H" in T else "",
                      "with four square legs"])
    return txt, b.proposal("four-leg table", out)


def chair(b: Builder):
    S = b.d("S", 350, 550, 450, "seat size")
    hs = b.d("hs", 350, 520, 450, "seat height")
    hb = b.d("hb", 250, 550, 400, "backrest height above the seat")
    b.derived("ts", round(S * 0.05, 1), "seat thickness")
    b.derived("lw", round(S * 0.08, 1), "leg size")
    b.derived("tb", round(S * 0.05, 1), "backrest thickness")
    b.op("seat", "box", size=["S", "S", "ts"], center=True)
    b.op("seat_p", "translate", input="seat", offset=[0, 0, "hs-ts/2"])
    b.op("leg", "box", size=["lw", "lw", "hs-ts+0.01[mm]"], center=True)
    b.op("leg0", "translate", input="leg", offset=["-S/2+lw/2", "-S/2+lw/2", "(hs-ts)/2"])
    b.op("legs_x", "linear_pattern", input="leg0", count=2, step=["S-lw", 0, 0])
    b.op("legs", "linear_pattern", input="legs_x", count=2, step=[0, "S-lw", 0])
    b.op("back", "box", size=["S", "tb", "hb"], center=True)
    b.op("back_p", "translate", input="back", offset=[0, "S/2-tb/2", "hs+hb/2-0.01[mm]"])
    out = b.op("body", "union", inputs=["seat_p", "legs", "back_p"])
    b.req("x", "S")
    T = b.t
    txt = b.sentence(b.pick(["chair", "simple chair", "dining chair", "wooden chair"]),
                     [f"{T['S']} square seat" if "S" in T else "", f"seat height {T['hs']}" if "hs" in T else "",
                      f"backrest {T['hb']} above the seat" if "hb" in T else ""])
    return txt, b.proposal("chair", out)


def ladder(b: Builder):
    H = b.d("H", 500, 4000, 2000, "height")
    W = b.d("W", 250, 600, 400, "outer width")
    n = b.n("n", 4, 14, 7, "rung count")
    rw = b.derived("rw", round(W * 0.12, 1), "rail width")
    b.derived("rd", round(W * 0.15, 1), "rail depth")
    b.derived("rr", round(W * 0.04, 1), "rung radius")
    if rw * 2 >= W - 50:
        return None
    b.op("rail", "box", size=["rw", "rd", "H"], center=True)
    b.op("rail0", "translate", input="rail", offset=["-(W-rw)/2", 0, "H/2"])
    b.op("rails", "linear_pattern", input="rail0", count=2, step=["W-rw", 0, 0])
    b.op("rung", "cylinder", radius="rr", height="W-rw", center=True)
    b.op("rung_r", "rotate", input="rung", axis=[0, 1, 0], angle=90)
    b.op("rung0", "translate", input="rung_r", offset=[0, 0, f"H/{n + 1}"])
    b.op("rungs", "linear_pattern", input="rung0", count=n, step=[0, 0, f"H/{n + 1}"])
    out = b.op("body", "union", inputs=["rails", "rungs"])
    b.req("x", "W")
    b.req("z", "H")
    T = b.t
    txt = b.sentence(b.pick(["ladder", "straight ladder", "step ladder section", "rung ladder"]),
                     [f"{T['H']} tall" if "H" in T else "", f"{T['W']} wide" if "W" in T else "",
                      f"{n} round rungs" if not b.vague else ""])
    return txt, b.proposal(f"ladder, {n} rungs", out)


def truss(b: Builder):
    L = b.d("L", 500, 6000, 2000, "span")
    Ht = b.d("Ht", 100, 1200, 400, "depth")
    n = b.n("n", 3, 10, 5, "panel count")
    s = b.derived("s", round(max(Ht * 0.08, 10), 1), "member size")
    p = L / n
    ang = round(math.degrees(math.atan2(Ht, p)), 4)
    b.op("chord", "box", size=["L", "s", "s"], center=True)
    b.op("bot", "translate", input="chord", offset=["L/2", 0, "s/2"])
    b.op("top", "translate", input="chord", offset=["L/2", 0, "Ht+s/2"])
    b.op("vert", "box", size=["s", "s", "Ht"], center=True)
    b.op("vert0", "translate", input="vert", offset=["s/2", 0, "Ht/2+s/2"])
    b.op("verts", "linear_pattern", input="vert0", count=n + 1, step=[f"(L-s)/{n}", 0, 0])
    b.op("diag", "box", size=[f"sqrt((L/{n})^2+Ht^2)", "s*0.8", "s*0.8"], center=True)
    b.op("diag_r", "rotate", input="diag", axis=[0, 1, 0], angle=-ang)
    b.op("diag0", "translate", input="diag_r", offset=[f"L/{n}/2", 0, "Ht/2+s/2"])
    b.op("diags", "linear_pattern", input="diag0", count=n, step=[f"L/{n}", 0, 0])
    b.op("frame", "union", inputs=["bot", "top", "verts", "diags"])
    b.op("clip", "box", size=["L", "4*s", "Ht+s"], center=False)
    b.op("clip_p", "translate", input="clip", offset=[0, "-2*s", 0])
    out = b.op("body", "intersection", inputs=["frame", "clip_p"])
    b.req("x", "L")
    T = b.t
    txt = b.sentence(b.pick(["planar truss", "Pratt truss", "bridge truss", "lattice girder"]),
                     [f"spanning {T['L']}" if "L" in T else "", f"{T['Ht']} deep" if "Ht" in T else "",
                      f"with {n} panels" if not b.vague else ""])
    return txt, b.proposal(f"truss, {n} panels", out)


def knob(b: Builder):
    D = b.d("D", 15, 80, 35, "diameter")
    H = b.d("H", 10, 50, 20, "height")
    n = b.n("n", 12, 40, 24, "grip groove count")
    b.derived("bd", round(max(D * 0.18, 3), 2), "shaft hole diameter")
    b.derived("gw", round(max(D * 0.04, 0.8), 2), "groove width")
    b.revolve_up("cap", [0, 0], [
        {"kind": "line", "to": ["D/2", 0]}, {"kind": "line", "to": ["D/2", "0.8*H"]},
        {"kind": "arc", "through": ["0.38*D", "0.96*H"], "to": [0, "H"]}, {"kind": "line", "to": [0, 0]}])
    b.op("g", "box", size=["2*gw", "gw", "1.6*H"], center=True)
    b.op("g0", "translate", input="g", offset=["D/2", 0, 0])
    b.op("grooves", "polar_pattern", input="g0", count=n)
    b.op("shaft", "cylinder", radius="bd/2", height="1.2*H", center=True)
    out = b.op("body", "difference", base="cap", tools=["grooves", "shaft"])
    T = b.t
    txt = b.sentence(b.pick(["control knob", "knurled knob", "volume knob", "grip knob"]),
                     [f"{T['D']} diameter" if "D" in T else "", f"{T['H']} tall" if "H" in T else "",
                      f"{n} grip grooves" if not b.vague else "", "domed top, with a shaft hole"])
    return txt, b.proposal("grooved knob", out)


def hinge_leaf(b: Builder):
    L = b.d("L", 30, 200, 80, "leaf length")
    W = b.d("W", 15, 80, 30, "leaf width")
    t = b.d("t", 1, 5, 2, "leaf thickness")
    k = b.n("k", 2, 4, 3, "knuckle count")
    b.derived("r", round(t * 1.6, 2), "knuckle radius")
    b.op("leaf", "box", size=["L", "W", "t"], center=True)
    b.op("leaf_p", "translate", input="leaf", offset=["L/2", "-W/2", "t/2"])
    seg = f"L/{2 * k - 1}"
    b.op("kn", "cylinder", radius="r", height=seg, center=True)
    b.op("kn_r", "rotate", input="kn", axis=[0, 1, 0], angle=90)
    b.op("kn0", "translate", input="kn_r", offset=[f"{seg}/2", 0, "r"])
    b.op("knuckles", "linear_pattern", input="kn0", count=k, step=[f"2*{seg}", 0, 0])
    b.op("solid", "union", inputs=["leaf_p", "knuckles"])
    b.op("pin", "cylinder", radius="r*0.45", height="3*L", center=True)
    b.op("pin_r", "rotate", input="pin", axis=[0, 1, 0], angle=90)
    b.op("pin_p", "translate", input="pin_r", offset=["L/2", 0, "r"])
    out = b.op("body", "difference", base="solid", tools=["pin_p"])
    b.req("x", "L")
    T = b.t
    txt = b.sentence(b.pick(["hinge leaf", "butt hinge half", "hinge plate", "door hinge leaf"]),
                     [f"{T['L']} long" if "L" in T else "", f"{T['W']} wide" if "W" in T else "",
                      f"{T['t']} thick" if "t" in T else "", f"{k} knuckles" if not b.vague else ""])
    return txt, b.proposal(f"hinge leaf, {k} knuckles", out)


# --------------------------------------------------------------------- open-ended feature grammar
def custom_part(b: Builder):
    """Base body + 1..4 features. The text is generated from the feature list, so any combination is covered."""
    base = b.pick(["block", "block", "disc", "plate"])
    if base == "disc":
        D = b.d("D", 30, 250, 100, "disc diameter")
        H = b.d("H", 5, 60, 20, "disc thickness")
        X = Y = D
        b.op("base", "cylinder", radius="D/2", height="H")
        noun = b.pick(["round blank", "disc", "cylindrical block", "puck"])
        head = [f"{b.t['D']} diameter" if "D" in b.t else "", f"{b.t['H']} thick" if "H" in b.t else ""]
    else:
        X = b.d("X", 30, 300, 100, "length")
        Y = b.d("Y", 20, min(X, 250), 60, "width")
        H = b.d("H", 3 if base == "plate" else 15, 15 if base == "plate" else 100, 10 if base == "plate" else 40,
                "height")
        b.op("base0", "box", size=["X", "Y", "H"], center=True)
        b.op("base", "translate", input="base0", offset=[0, 0, "H/2"])
        noun = b.pick(["rectangular block", "machined block", "plate", "base part"])
        head = [f"{b.t['X']} x {b.t['Y']}" if "X" in b.t else "", f"{b.t['H']} high" if "H" in b.t else ""]
    cuts, adds, text = [], [], []
    feats = list(b.rng.choice(["hole", "holes4", "pocket", "boss", "slot", "cbore", "polar"],
                              size=int(b.rng.integers(1, 5)), replace=False))
    m = min(X, Y)
    for i, f in enumerate(feats):
        k = f"f{i}"
        if f == "hole":
            b.derived(f"{k}d", round(m * 0.2, 2), "centre hole diameter")
            cuts.append(b.op(k, "cylinder", radius=f"{k}d/2", height="3*H", center=True))
            text.append("a through hole in the centre")
        elif f == "holes4" and base != "disc":
            b.derived(f"{k}d", round(m * 0.1, 2), "corner hole diameter")
            b.derived(f"{k}e", round(m * 0.15, 2), "corner hole inset")
            b.op(f"{k}c", "cylinder", radius=f"{k}d/2", height="3*H", center=True)
            b.op(f"{k}c0", "translate", input=f"{k}c", offset=[f"-X/2+{k}e", f"-Y/2+{k}e", 0])
            b.op(f"{k}r", "linear_pattern", input=f"{k}c0", count=2, step=[f"X-2*{k}e", 0, 0])
            cuts.append(b.op(k, "linear_pattern", input=f"{k}r", count=2, step=[0, f"Y-2*{k}e", 0]))
            text.append("a hole near each corner")
        elif f == "pocket":
            b.derived(f"{k}w", round(m * 0.45, 2), "pocket size")
            b.derived(f"{k}h", round(H * 0.4, 2), "pocket depth")
            b.op(f"{k}b", "box", size=[f"{k}w", f"{k}w", f"2*{k}h"], center=True)
            cuts.append(b.op(k, "translate", input=f"{k}b", offset=[0, 0, "H"]))
            text.append("a square pocket in the top")
        elif f == "boss":
            b.derived(f"{k}d", round(m * 0.35, 2), "boss diameter")
            b.derived(f"{k}h", round(H * 0.5, 2), "boss height")
            b.op(f"{k}c", "cylinder", radius=f"{k}d/2", height=f"{k}h+0.01[mm]")
            adds.append(b.op(k, "translate", input=f"{k}c", offset=[0, 0, "H-0.01[mm]"]))
            text.append("a round boss on top")
        elif f == "slot" and base != "disc":
            b.derived(f"{k}w", round(m * 0.12, 2), "slot width")
            b.op(f"{k}b", "box", size=["X*0.5", f"{k}w", "3*H"], center=True)
            cuts.append(b.op(k, "translate", input=f"{k}b", offset=[0, "Y*0.28", 0]))
            text.append("a through slot along the length")
        elif f == "cbore":
            b.derived(f"{k}d", round(m * 0.14, 2), "counterbored hole diameter")
            b.op(f"{k}a", "cylinder", radius=f"{k}d/2", height="3*H", center=True)
            b.op(f"{k}b", "cylinder", radius=f"{k}d", height="H*0.6", center=True)
            b.op(f"{k}bp", "translate", input=f"{k}b", offset=[0, 0, "H"])
            b.op(f"{k}u", "union", inputs=[f"{k}a", f"{k}bp"])
            cuts.append(b.op(k, "translate", input=f"{k}u", offset=["-X*0.3", 0, 0]))
            text.append("a counterbored hole off-centre")
        elif f == "polar":
            nn = int(b.rng.integers(3, 9)) if not b.vague else 6
            b.derived(f"{k}d", round(m * 0.07, 2), "pattern hole diameter")
            b.op(f"{k}c", "cylinder", radius=f"{k}d/2", height="3*H", center=True)
            b.op(f"{k}c0", "translate", input=f"{k}c", offset=[f"{m * 0.32:.3f}", 0, 0])
            cuts.append(b.op(k, "polar_pattern", input=f"{k}c0", count=nn))
            text.append(f"{nn} holes on a bolt circle")
    if not text:
        return None
    cur = "base"
    if adds:
        cur = b.op("added", "union", inputs=[cur, *adds])
    out = b.op("body", "difference", base=cur, tools=cuts) if cuts else cur
    if base != "disc":
        b.req("x", "X")
        b.req("y", "Y")
    else:
        b.req("x", "D")
    if not adds:
        b.req("z", "H")
    txt = b.sentence(noun, head + ["with " + ", ".join(text[:-1]) + (" and " if len(text) > 1 else "") + text[-1]])
    return txt, b.proposal(f"{noun} with " + ", ".join(feats), out)


FAMILIES: dict[str, Callable] = {f.__name__: f for f in (
    plate_holes, l_bracket, u_channel, i_beam, flange, washer, bushing, stepped_shaft, hex_nut, hex_bolt, pulley,
    spoked_wheel, pillow_block, sprocket, heat_sink, enclosure, honeycomb_panel, pipe_tee, pipe_elbow,
    pressure_vessel, funnel, rocket_nozzle, rocket_body, propeller, mug, bottle, vase, table, chair, ladder, truss,
    knob, hinge_leaf, custom_part)}

# Never trained on: they measure generalisation to unseen object kinds built from the same vocabulary.
HELDOUT_FAMILIES = ("pipe_elbow", "chair", "knob", "hinge_leaf")


def sample(family: str, seed: int, vague: bool):
    rng = np.random.default_rng(seed)
    b = Builder(rng, vague)
    try:
        r = FAMILIES[family](b)
    except (ValueError, ZeroDivisionError):
        return None
    return r
