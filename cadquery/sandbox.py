"""Run an untrusted CadQuery script and return its geometry: STEP bytes, GLB bytes, volume, bounding box, or the error.

Two layers, both always on:

1. **Static check** (``check_script``): the script is parsed and walked. Only ``import cadquery``/``import math``
   (and ``from`` either) are allowed; any name or attribute that starts with ``_`` (dunders included) is rejected,
   as are the builtins and CadQuery members that reach the file system, the interpreter or other code
   (``open``, ``exec``, ``eval``, ``compile``, ``__import__``, ``getattr``, ``exporters``, ``importers``, ``save``,
   ``wrapped`` ...). The script must assign ``result`` at module level.
2. **Process isolation** (``run_script``): a fresh ``python -I`` child with an empty environment (no tokens),
   ``RLIMIT_AS`` 4 GB, ``RLIMIT_CPU`` 30 s, a 30 s wall clock (process group killed), a size cap on written files,
   its own temp directory, restricted builtins, ``socket`` disabled, and no network namespace when
   ``unshare -rn`` works on the host (recorded in the result as ``net_isolation``).

The geometry numbers come from OCCT (``Shape.Volume``, ``BoundingBox``, ``isValid``) inside the child; nothing the
model wrote is reported as a measurement. Usable as a module or ``python cadquery/sandbox.py script.py``."""

from __future__ import annotations

import ast
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

TIMEOUT_S = 30
MEM_BYTES = 4 * 1024 ** 3
FILE_BYTES = 256 * 1024 ** 2
ALLOWED_MODULES = {"cadquery", "math"}
DENY_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "input", "breakpoint", "help", "memoryview", "classmethod", "staticmethod", "property", "super", "type",
    "object", "dir", "id", "hash", "exit", "quit", "copyright", "credits", "license", "os", "sys", "subprocess",
    "socket", "shutil", "pathlib", "importlib", "builtins", "ctypes", "pickle", "marshal", "io", "threading",
    "multiprocessing", "signal", "resource", "urllib", "http", "requests", "OCP", "cq_editor", "show_object",
    "debug", "Path", "file", "reload", "execfile", "bytearray",
}
DENY_ATTRS = {
    "exporters", "importers", "export", "exportStep", "exportStl", "exportBrep", "exportSvg", "exportDXF",
    "importStep", "importBrep", "importDXF", "importStl", "save", "occ_impl", "cqgi", "vis", "wrapped",
    "OCP", "jupyter_tools", "selectors_ext", "fromBrep", "toOCC", "system", "popen", "remove", "unlink",
    "rmdir", "mkdir", "write", "read", "load", "loads", "dump", "dumps", "func_globals", "gi_frame", "f_globals",
    "f_locals", "f_builtins", "tb_frame", "cr_frame", "ag_frame",
}
SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "int",
    "isinstance", "len", "list", "map", "max", "min", "pow", "print", "range", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None", "ArithmeticError",
    "AssertionError", "Exception", "IndexError", "KeyError", "RuntimeError", "TypeError", "ValueError",
    "ZeroDivisionError", "StopIteration",
)


@dataclass
class SandboxResult:
    ok: bool
    stage: str                       # rejected | timeout | memory | runtime | no_result | invalid | ok
    error: str = ""
    volume: float | None = None      # mm^3, OCCT
    bbox: list[float] | None = None  # [xmin, ymin, zmin, xmax, ymax, zmax] mm, OCCT
    n_solids: int = 0
    valid: bool = False              # OCCT BRepCheck on the result shape
    step: bytes | None = field(default=None, repr=False)
    glb: bytes | None = field(default=None, repr=False)
    seconds: float = 0.0
    net_isolation: str = ""

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("step"), d.pop("glb")
        d["step_bytes"] = len(self.step or b"")
        d["glb_bytes"] = len(self.glb or b"")
        return d


# --------------------------------------------------------------------------------------------- static check
def check_script(src: str) -> list[str]:
    """All reasons the script is not allowed to run (empty list = allowed)."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]
    bad: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_MODULES or "." in a.name:
                    bad.append(f"line {line}: import {a.name} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "") not in ALLOWED_MODULES:
                bad.append(f"line {line}: from {node.module} import is not allowed")
            for a in node.names:
                if a.name == "*" or a.name.startswith("_") or a.name in DENY_NAMES or a.name in DENY_ATTRS:
                    bad.append(f"line {line}: importing {a.name} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in DENY_NAMES:
                bad.append(f"line {line}: name {node.id} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in DENY_ATTRS:
                bad.append(f"line {line}: attribute .{node.attr} is not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                bad.append(f"line {line}: definition {node.name} is not allowed")
            if isinstance(node, ast.ClassDef):
                bad.append(f"line {line}: class definitions are not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith, ast.Global,
                               ast.Nonlocal)):
            bad.append(f"line {line}: {type(node).__name__} is not allowed")
        elif isinstance(node, ast.alias) and node.asname and node.asname.startswith("_"):
            bad.append(f"line {line}: alias {node.asname} is not allowed")
        elif isinstance(node, ast.arg) and node.arg.startswith("_"):
            bad.append(f"line {line}: argument {node.arg} is not allowed")
        elif isinstance(node, ast.keyword) and node.arg and node.arg.startswith("_"):
            bad.append(f"line {line}: keyword {node.arg} is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            bad.append(f"line {line}: string containing '__' is not allowed")
    if not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
               for n in tree.body):
        bad.append("the script must assign `result = ...` at module level")
    return bad


# --------------------------------------------------------------------------------------------- child process
CHILD = r'''
import base64, json, os, struct, sys, time
def run_one(req):
    out = {"stage": "runtime"}
    try:
        import socket
        def _no_net(*a, **k):
            raise OSError("network disabled in the sandbox")
        socket.socket = _no_net
        socket.create_connection = _no_net
        socket.getaddrinfo = _no_net
        import builtins, math
        import cadquery as cq
        safe = {k: getattr(builtins, k) for k in req["builtins"] if hasattr(builtins, k)}
        safe["__import__"] = _limited_import
        g = {"__builtins__": safe, "__name__": "berkelium_cq_script"}
        exec(compile(req["src"], "<script>", "exec"), g)
        if "result" not in g:
            out.update(stage="no_result", error="script did not set `result`")
            return out
        shape = to_shape(cq, g["result"])
        if shape is None:
            out.update(stage="no_result", error=f"`result` is {type(g['result']).__name__}, not CadQuery geometry")
            return out
        solids = shape.Solids()
        bb = shape.BoundingBox()
        out.update(volume=float(shape.Volume()), bbox=[bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax],
                   n_solids=len(solids), valid=bool(shape.isValid()))
        if not solids or out["volume"] <= 0:
            out.update(stage="invalid", error=f"result has {len(solids)} solids and volume {out['volume']:.6g}")
            return out
        step = req["tmp"] + "/result.step"
        cq.exporters.export(shape, step, exportType="STEP")
        out["step_b64"] = base64.b64encode(open(step, "rb").read()).decode()
        verts, tris = shape.tessellate(req["tol"], 0.2)
        out["glb_b64"] = base64.b64encode(glb([(v.x, v.y, v.z) for v in verts], tris)).decode()
        out["stage"] = "ok" if out["valid"] else "invalid"
        if not out["valid"]:
            out["error"] = "OCCT BRepCheck reports the shape as invalid"
    except MemoryError:
        out.update(stage="memory", error="memory limit (4 GB) exceeded")
    except BaseException as e:
        out.update(stage="runtime", error=f"{type(e).__name__}: {e}"[:2000])
    return out

def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ("cadquery", "math"):
        raise ImportError(f"import of {name} is not allowed")
    import importlib
    return importlib.import_module(name)

def to_shape(cq, r):
    if isinstance(r, cq.Workplane):
        vals = [v for v in r.vals() if isinstance(v, cq.Shape)]
        if not vals:
            return None
        return vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
    if isinstance(r, cq.Assembly):
        return r.toCompound()
    if isinstance(r, cq.Shape):
        return r
    if isinstance(r, (list, tuple)) and r and all(isinstance(x, cq.Shape) for x in r):
        return cq.Compound.makeCompound(list(r))
    return None

def glb(verts, tris):
    import array
    pos = array.array("f", [c for v in verts for c in v]).tobytes()
    idx = array.array("I", [i for t in tris for i in t]).tobytes()
    xs = [v[0] for v in verts] or [0]; ys = [v[1] for v in verts] or [0]; zs = [v[2] for v in verts] or [0]
    gltf = {"asset": {"version": "2.0", "generator": "berkelium-cq-sandbox"}, "scene": 0,
            "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "buffers": [{"byteLength": len(pos) + len(idx)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos), "target": 34962},
                            {"buffer": 0, "byteOffset": len(pos), "byteLength": len(idx), "target": 34963}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(verts), "type": "VEC3",
                           "min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]},
                          {"bufferView": 1, "componentType": 5125, "count": 3 * len(tris), "type": "SCALAR"}]}
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    bin_ = pos + idx
    bin_ += b"\0" * (-len(bin_) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(bin_), 0x004E4942) + bin_
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body

def serve():
    """Fork server: cadquery is imported once; every script runs in its own forked child with rlimits."""
    import resource, select, signal
    import cadquery  # noqa: F401  (warm import, shared copy-on-write with every child)
    for line in sys.stdin:
        req = json.loads(line)
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (req["mem"], req["mem"]))
                resource.setrlimit(resource.RLIMIT_CPU, (req["cpu"], req["cpu"] + 1))
                tmp = req["tmp"] + "/" + str(os.getpid())
                os.mkdir(tmp)
                req["tmp"] = tmp
                res = run_one(req)
            except BaseException as e:
                res = {"stage": "runtime", "error": f"{type(e).__name__}: {e}"}
            with os.fdopen(w, "w") as f:
                f.write(json.dumps(res))
            os._exit(0)
        os.close(w)
        chunks, deadline, timed_out = [], time.time() + req["timeout"], False
        while True:
            left = deadline - time.time()
            if left <= 0:
                timed_out = True
                break
            rd, _, _ = select.select([r], [], [], left)
            if not rd:
                continue
            b = os.read(r, 1 << 20)
            if not b:
                break
            chunks.append(b)
        if timed_out:
            os.kill(pid, signal.SIGKILL)
        _, status = os.waitpid(pid, 0)
        os.close(r)
        txt = b"".join(chunks).decode(errors="replace")
        if timed_out:
            res = {"stage": "timeout", "error": f"wall-clock limit ({req['timeout']:g} s) exceeded"}
        elif not txt:
            sig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else 0
            res = {"stage": "timeout" if sig == signal.SIGXCPU else "memory" if sig in (signal.SIGKILL, signal.SIGSEGV, signal.SIGABRT) else "runtime",
                   "error": f"script process died (signal {sig}, status {status})"}
        else:
            res = json.loads(txt)
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if len(sys.argv) > 1 and sys.argv[1] == "serve":
    serve()
else:
    res = run_one(json.loads(sys.stdin.read()))
    sys.stdout.write("\n@@BERKELIUM_CQ@@" + json.dumps(res))
'''


def _limits(mem: int, cpu: int):
    def f():
        import resource
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return f


_NET: str | None = None


def net_isolation() -> str:
    """'netns' when the child can get its own empty network namespace, else 'socket-disabled'."""
    global _NET
    if _NET is None:
        _NET = "socket-disabled"
        if shutil.which("unshare") and os.environ.get("BERKELIUM_SANDBOX_NETNS", "1") != "0":
            try:
                if subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10).returncode == 0:
                    _NET = "netns"
            except (OSError, subprocess.SubprocessError):
                pass
    return _NET


def run_script(src: str, timeout: float = TIMEOUT_S, mem_bytes: int = MEM_BYTES, tol: float = 0.1,
               python: str | None = None) -> SandboxResult:
    """Static check, then run in an isolated child. Never raises for a bad script."""
    t0 = time.time()
    bad = check_script(src)
    if bad:
        return SandboxResult(False, "rejected", "; ".join(bad)[:2000], seconds=round(time.time() - t0, 3))
    net = net_isolation()
    tmp = tempfile.mkdtemp(prefix="bcq_")
    cmd = [python or sys.executable, "-I", "-c", CHILD]
    if net == "netns":
        cmd = ["unshare", "-rn", *cmd]
    env = _child_env(tmp)
    req = json.dumps({"src": src, "tmp": tmp, "tol": tol, "builtins": SAFE_BUILTINS})
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=tmp, env=env, preexec_fn=_limits(mem_bytes, int(timeout)))
        try:
            so, se = p.communicate(req.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.communicate()
            return SandboxResult(False, "timeout", f"wall-clock limit ({timeout:g} s) exceeded",
                                 seconds=round(time.time() - t0, 3), net_isolation=net)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    dt = round(time.time() - t0, 3)
    text = so.decode(errors="replace")
    if "@@BERKELIUM_CQ@@" not in text:
        err = se.decode(errors="replace")[-1500:]
        stage = "memory" if "MemoryError" in err or p.returncode in (-9, -6) and "alloc" in err else \
            "timeout" if p.returncode in (-signal.SIGXCPU, -signal.SIGKILL) else "runtime"
        return SandboxResult(False, stage, f"child exited {p.returncode}: {err.strip()[-800:]}", seconds=dt,
                             net_isolation=net)
    return _decode(json.loads(text.rsplit("@@BERKELIUM_CQ@@", 1)[1]), dt, net)


def _child_env(tmp: str) -> dict:
    env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "OMP_NUM_THREADS": "1",
           "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    if os.environ.get("LD_LIBRARY_PATH"):          # OCCT needs libGL from the env on some hosts; no secrets here
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    return env


def _decode(r: dict, dt: float, net: str) -> SandboxResult:
    step = base64.b64decode(r["step_b64"]) if r.get("step_b64") else None
    glb = base64.b64decode(r["glb_b64"]) if r.get("glb_b64") else None
    return SandboxResult(r["stage"] == "ok", r["stage"], r.get("error", ""), r.get("volume"), r.get("bbox"),
                         r.get("n_solids", 0), r.get("valid", False), step, glb, dt, net)


class BatchRunner:
    """Many scripts, same guarantees: ``jobs`` fork servers (each in its own network namespace when available,
    empty environment), every script statically checked and then run in a fresh forked child with the 4 GB /
    30 s limits. ``keep_artifacts=False`` drops STEP/GLB bytes (data building only needs the numbers)."""

    def __init__(self, jobs: int | None = None, timeout: float = TIMEOUT_S, mem_bytes: int = MEM_BYTES,
                 tol: float = 0.1, keep_artifacts: bool = False, python: str | None = None):
        import queue
        import threading
        self.timeout, self.mem, self.tol, self.keep = timeout, mem_bytes, tol, keep_artifacts
        self.net = net_isolation()
        self.tmp = tempfile.mkdtemp(prefix="bcq_batch_")
        cmd = [python or sys.executable, "-I", "-c", CHILD, "serve"]
        if self.net == "netns":
            cmd = ["unshare", "-rn", *cmd]
        self._free: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._cmd = cmd
        for _ in range(jobs or os.cpu_count() or 1):
            self._free.put(self._spawn())

    def _spawn(self):
        def pre():
            os.setsid()
            import resource
            resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        return subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                cwd=self.tmp, env=_child_env(self.tmp), text=True, bufsize=1, preexec_fn=pre)

    def run(self, src: str) -> SandboxResult:
        t0 = time.time()
        bad = check_script(src)
        if bad:
            return SandboxResult(False, "rejected", "; ".join(bad)[:2000], seconds=round(time.time() - t0, 3))
        p = self._free.get()
        try:
            p.stdin.write(json.dumps({"src": src, "tmp": self.tmp, "tol": self.tol, "builtins": SAFE_BUILTINS,
                                      "mem": self.mem, "cpu": int(self.timeout), "timeout": self.timeout}) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
            if not line:
                raise BrokenPipeError("fork server died")
            r = json.loads(line)
        except (BrokenPipeError, OSError, json.JSONDecodeError) as e:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            p = self._spawn()
            r = {"stage": "runtime", "error": f"sandbox server failure: {e}"}
        finally:
            self._free.put(p)
        if not self.keep:
            r.pop("step_b64", None), r.pop("glb_b64", None)
        return _decode(r, round(time.time() - t0, 3), self.net)

    def map(self, srcs, jobs: int | None = None):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(jobs or self._free.qsize() or 1) as ex:
            yield from ex.map(self.run, srcs)

    def close(self):
        while not self._free.empty():
            p = self._free.get()
            try:
                p.stdin.close()
                p.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                p.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("script")
    ap.add_argument("--step", help="write the STEP file here")
    ap.add_argument("--glb", help="write the GLB file here")
    a = ap.parse_args()
    r = run_script(Path(a.script).read_text())
    if a.step and r.step:
        Path(a.step).write_bytes(r.step)
    if a.glb and r.glb:
        Path(a.glb).write_bytes(r.glb)
    print(json.dumps(r.summary(), indent=1))
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
