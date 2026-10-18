"""Berkelium procedural Geometry IR (as written by services/neural/berkelium_neural/cad/families.py) -> CadQuery.

The generated script follows the core's OCCT backend op for op (berkelium/geometry/occt.py @ 737dc35), so the
CadQuery solid matches the core's solid:

* parameters become named Python variables in mm / degrees / counts; op arguments stay expressions over them
  (``^`` -> ``**``, ``1[mm]`` -> ``1``, ``sqrt`` -> ``math.sqrt``); an expression that cannot be translated is
  emitted as its numeric value (core-resolved when the core is installed);
* box (corner at the origin unless centred), cylinder/cone/torus on +Z from the origin, profiles in the XY plane,
  extrude along +Z, revolve about the Y axis, rotate about an axis through ``origin``, patterns as sequential
  fuses, booleans folded left to right;
* splines use the core's own curve samples (``berkelium.geometry.curves.spline_points``), so the curve is the
  same one the core builds.

``scripts/cq_data.py`` keeps a transpiled row only when the sandboxed CadQuery solid and the core's solid agree
on volume and bounding box (tests: cadquery/tests/test_transpile.py)."""

from __future__ import annotations

import keyword
import math
import re

RESERVED = {"cq", "math", "result", "k"} | set(keyword.kwlist) | {
    "abs", "min", "max", "round", "range", "len", "sum", "float", "int", "list", "print", "pow"}
_FUNCS = {"sqrt": "math.sqrt", "sin": "math.sin", "cos": "math.cos", "tan": "math.tan", "atan": "math.atan",
          "atan2": "math.atan2", "asin": "math.asin", "acos": "math.acos", "exp": "math.exp", "log": "math.log",
          "floor": "math.floor", "ceil": "math.ceil", "abs": "abs", "min": "min", "max": "max"}
_ANGLE_FUNCS = {"sin", "cos", "tan"}           # core takes these on dimensioned angles; leave numeric


class TranspileError(ValueError):
    pass


def _num(x: float) -> str:
    x = float(x)
    if x == int(x) and abs(x) < 1e12:
        return str(int(x))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _ident(name: str, taken: set[str]) -> str:
    n = re.sub(r"\W", "_", name)
    if not n or n[0].isdigit() or n.startswith("_"):
        n = "v" + n.lstrip("_")
    while n in RESERVED or n in taken:
        n += "_"
    return n


class _Expr:
    def __init__(self, names: dict[str, str], env: dict[str, float]):
        self.names, self.env = names, env

    def py(self, v, resolved=None) -> str:
        """Python source for one scalar argument. ``resolved`` is the core's value (used as the fallback)."""
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            return _num(v)
        if not isinstance(v, str):
            raise TranspileError(f"unsupported argument {v!r}")
        src = v.strip()
        if re.search(r"\[(?!mm\])[^\]]*\]", src) or any(re.search(rf"\b{f}\s*\(", src) for f in _ANGLE_FUNCS):
            return self._fallback(v, resolved)
        src = re.sub(r"\[mm\]", "", src).replace("^", "**")

        def name(m):
            t = m.group(0)
            if t in _FUNCS:
                return _FUNCS[t]
            if t == "pi":
                return "math.pi"
            if t in self.names:
                return self.names[t]
            raise TranspileError(f"unknown name {t!r} in {v!r}")
        try:
            out = re.sub(r"(?<![\w.])[A-Za-z_]\w*", name, src)
            val = eval(out, {"__builtins__": {"abs": abs, "min": min, "max": max}, "math": math},
                       {self.names[k]: x for k, x in self.env.items()})
        except TranspileError:
            raise
        except Exception:
            return self._fallback(v, resolved)
        if resolved is not None and not math.isclose(float(val), float(resolved), rel_tol=1e-9, abs_tol=1e-9):
            return self._fallback(v, resolved)
        return out

    def _fallback(self, v, resolved):
        if resolved is None:
            raise TranspileError(f"cannot translate {v!r} and no core-resolved value is available")
        return _num(resolved)

    def vec(self, vs, resolved=None) -> str:
        r = resolved if resolved is not None else [None] * len(vs)
        return "(" + ", ".join(self.py(a, b) for a, b in zip(vs, r)) + ")"


def _param_value(p: dict) -> float:
    v = p.get("value")
    if isinstance(v, dict):
        v = v.get("value")
    return float(v)


def core_resolved(proposal: dict, component: str | None = None) -> list[dict] | None:
    """The core's resolved op list for the procedural component (None when the core is not installed)."""
    try:
        from berkelium.pipeline import run
    except ImportError:
        return None
    res = run(proposal, realize=False)
    if res.record is None:
        return None
    for rc in res.record.realized:
        if component is None or rc.component == component:
            if rc.geometry is not None:
                return [o.model_dump() for o in rc.geometry.ops]
    return None


def transpile(proposal: dict, resolved_ops: list[dict] | None = None, use_core: bool = True,
              header_comment: bool = True) -> str:
    """CadQuery source for the first procedural component of ``proposal``; ``result`` is its output solid."""
    comps = [c for c in proposal["structure"]["components"] if c.get("kind") == "procedural"]
    if not comps:
        raise TranspileError("no procedural component")
    comp = comps[0]
    g = comp["geometry"]
    if g.get("units", "mm") != "mm":
        raise TranspileError("only mm geometry is supported")
    if resolved_ops is None and use_core:
        resolved_ops = core_resolved(proposal, comp["id"])
    rmap = {o["id"]: o for o in resolved_ops} if resolved_ops else {}
    params = proposal.get("specification", {}).get("parameters", [])
    names: dict[str, str] = {}
    env: dict[str, float] = {}
    taken: set[str] = set()
    lines = ["import cadquery as cq", "import math", ""]
    for p in params:
        if p.get("kind") not in ("length", "angle", "count", "real"):
            continue
        try:
            val = _param_value(p)
        except (TypeError, ValueError):
            continue
        if p.get("kind") == "length" and p.get("unit", "mm") != "mm":
            raise TranspileError(f"parameter {p['id']} is not in mm")
        n = _ident(p["id"], taken)
        taken.add(n)
        names[p["id"]], env[p["id"]] = n, val
        unit = {"length": "  # mm", "angle": "  # deg"}.get(p["kind"], "")
        lines.append(f"{n} = {int(val) if p['kind'] == 'count' else _num(val)}{unit}")
    lines.append("")
    E = _Expr(names, env)
    var: dict[str, str] = {}

    def V(oid):
        if oid not in var:
            raise TranspileError(f"op refers to {oid!r} before it is defined")
        return var[oid]
    for o in g["ops"]:
        oid, kind = o["id"], o["op"]
        r = rmap.get(oid, {})
        n = _ident(oid, taken)
        taken.add(n)
        var[oid] = n

        def a(key, default=None):
            return E.py(o.get(key, default), r.get(key))
        if kind == "box":
            sz = E.vec(o["size"], r.get("size"))
            c = bool(o.get("center", False))
            lines.append(f"{n} = cq.Solid.makeBox(*{sz}" + (")" if not c else
                         f", pnt=cq.Vector(*{sz}) * -0.5)"))
        elif kind == "cylinder":
            rad, h = a("radius"), a("height")
            pnt = f", pnt=cq.Vector(0, 0, -({h}) / 2)" if o.get("center") else ""
            lines.append(f"{n} = cq.Solid.makeCylinder({rad}, {h}{pnt})")
        elif kind == "cone":
            lines.append(f"{n} = cq.Solid.makeCone({a('radius1')}, {a('radius2')}, {a('height')})")
        elif kind == "sphere":
            lines.append(f"{n} = cq.Solid.makeSphere({a('radius')}, angleDegrees1=-90, angleDegrees2=90)")
        elif kind == "torus":
            lines.append(f"{n} = cq.Solid.makeTorus({a('major_radius')}, {a('minor_radius')})")
        elif kind == "circle":
            c = E.vec(o.get("center", [0, 0]), r.get("center"))
            lines.append(f"{n} = cq.Workplane(\"XY\").moveTo(*{c}).circle({a('radius')})")
        elif kind == "profile":
            lines.append(_profile(n, o, r, E))
        elif kind == "extrude":
            lines.append(f"{n} = {V(o['profile'])}.extrude({a('height')}).val()")
        elif kind == "revolve":
            ang = a("angle", 360.0)
            lines.append(f"{n} = {V(o['profile'])}.revolve({ang}, (0, 0, 0), (0, 1, 0)).val()")
        elif kind in ("union", "intersection"):
            m = "fuse" if kind == "union" else "intersect"
            ins = [V(i) for i in o["inputs"]]
            lines.append(f"{n} = {ins[0]}" + "".join(f".{m}({i})" for i in ins[1:]))
        elif kind == "difference":
            lines.append(f"{n} = {V(o['base'])}" + "".join(f".cut({V(t)})" for t in o["tools"]))
        elif kind == "translate":
            lines.append(f"{n} = {V(o['input'])}.translate(cq.Vector(*{E.vec(o['offset'], r.get('offset'))}))")
        elif kind == "rotate":
            org = E.vec(o.get("origin", [0, 0, 0]), r.get("origin"))
            ax = E.vec(o.get("axis", [0, 0, 1]), r.get("axis"))
            lines.append(f"{n} = {V(o['input'])}.rotate(cq.Vector(*{org}), cq.Vector(*{org}) + cq.Vector(*{ax}), "
                         f"{a('angle', 0.0)})")
        elif kind == "mirror":
            org = E.vec(o.get("origin", [0, 0, 0]), r.get("origin"))
            lines.append(f"{n} = {V(o['input'])}.mirror(cq.Vector(*{E.vec(o['normal'], r.get('normal'))}), "
                         f"cq.Vector(*{org}))")
        elif kind == "linear_pattern":
            cnt = int(o["count"])
            st = E.vec(o["step"], r.get("step"))
            src = V(o["input"])
            lines.append(f"{n} = {src}")
            if cnt > 1:
                lines.append(f"for k in range(1, {cnt}):\n    {n} = {n}.fuse({src}.translate(cq.Vector(*{st}) * k))")
        elif kind == "polar_pattern":
            cnt = int(o["count"])
            tot = a("angle", 360.0)
            src = V(o["input"])
            full = abs(float(r.get("angle", o.get("angle", 360.0)) if not isinstance(o.get("angle", 360.0), str)
                             else r.get("angle", 360.0)) - 360.0) < 1e-9
            stepx = f"({tot}) / {cnt}" if full else (f"({tot}) / {cnt - 1}" if cnt > 1 else "0")
            lines.append(f"{n} = {src}")
            if cnt > 1:
                lines.append(f"for k in range(1, {cnt}):\n    {n} = {n}.fuse({src}.rotate(cq.Vector(0, 0, 0), "
                             f"cq.Vector(0, 0, 1), {stepx} * k))")
        else:
            raise TranspileError(f"op {kind!r} has no CadQuery translation")
    out = g["outputs"]
    body = out.get("body") or next(iter(out.values()))
    lines += ["", f"result = {V(body)}"]
    src = "\n".join(lines) + "\n"
    if header_comment:
        src = f"# {proposal.get('intent', {}).get('summary', comp['id'])}\n" + src
    return src


def _profile(n: str, o: dict, r: dict, E: _Expr) -> str:
    from_core = r.get("segments")
    start = E.vec(o["start"], r.get("start"))
    parts = [f"{n} = (cq.Workplane(\"XY\").moveTo(*{start})"]
    cur = _point(o["start"], r.get("start"), E)
    for i, seg in enumerate(o["segments"]):
        rs = from_core[i] if from_core else {}
        if seg["kind"] == "line":
            parts.append(f"    .lineTo(*{E.vec(seg['to'], rs.get('to'))})")
            cur = _point(seg["to"], rs.get("to"), E)
        elif seg["kind"] == "arc":
            parts.append(f"    .threePointArc({E.vec(seg['through'], rs.get('through'))}, "
                         f"{E.vec(seg['to'], rs.get('to'))})")
            cur = _point(seg["to"], rs.get("to"), E)
        elif seg["kind"] == "spline":
            from berkelium.geometry.curves import spline_points
            pts = [_point(p, rp, E) for p, rp in zip(seg["points"], rs.get("points") or [None] * len(seg["points"]))]
            dense = spline_points([cur] + pts)
            parts.append("    .spline([" + ", ".join(f"({_num(round(x, 6))}, {_num(round(y, 6))})"
                                                    for x, y in dense[1:]) + "], includeCurrent=True)")
            cur = pts[-1]
        else:
            raise TranspileError(f"segment {seg['kind']!r} has no CadQuery translation")
    parts.append("    .close())")
    return "\n".join(parts)


def _point(p, rp, E: _Expr) -> tuple[float, float]:
    """Numeric value of a 2-D point (core-resolved when given)."""
    if rp is not None:
        return float(rp[0]), float(rp[1])
    env = {E.names[k]: v for k, v in E.env.items()}
    return tuple(float(eval(E.py(c), {"__builtins__": {"abs": abs, "min": min, "max": max}, "math": math}, env))
                 for c in p)
