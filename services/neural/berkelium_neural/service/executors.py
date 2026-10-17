"""Allowlisted job executors: text_to_3d and render. Each stage is idempotent and cancellable; the optional
Gaussian fitting stage checkpoints to the job workdir and resumes after a restart."""

from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import __version__
from ..contracts import ArtifactKind, RenderConfig, RenderRequest, TextTo3DRequest
from .artifacts import ArtifactStore
from .jobs import Cancelled, JobContext

REPO = Path(__file__).resolve().parents[4]
PKG_ROOT = Path(__file__).resolve().parents[2]          # services/neural (holds the berkelium_neural package)
OPEN_WORLD_LABEL = "model-generated, normalized units, not measured geometry"
PIPELINES = {"shap-e-text300m": "text->CLIP ViT-L/14->Shap-E latent->STF mesh",
             "zimage-triposr": "text->canonical anchor prompt->Z-Image-Turbo anchor->colour key->TripoSR->clean mesh",
             "berkelium-textshape": "text->SDF->marching_cubes",
             "cad": "text->Qwen3-8B design compiler->DesignProposal->Berkelium core (L0-L6 validation)->"
                    "exact BREP (OCCT) / Manifold mesh",
             "cadquery": "text->E-C2 compiler (Qwen3 + CadQuery LoRA)->CadQuery script->sandbox->STEP/GLB->"
                         "Berkelium core validation (repair <= 2 rounds)->mesh"}
SHAP_E_MIN_FREE_MIB = 2000          # E-T3: worker peak ~1.5 GB RSS (1.1 GB anonymous) + ~0.8 GB watchdog margin


class ModelUnavailable(RuntimeError):
    code = "MODEL_UNAVAILABLE"


def model_registry() -> dict:
    """name -> {checkpoint, adapters{name: path}}. Override with BERKELIUM_MODELS (JSON). Paths are resolved
    relative to the repository root; nothing is downloaded."""
    if os.environ.get("BERKELIUM_MODELS"):
        reg = json.loads(os.environ["BERKELIUM_MODELS"])
    else:
        # The released checkpoints committed under exports/ (sha256 in each MODEL_CARD.json).
        reg = {"berkelium-textshape": {"checkpoint": "exports/textshape-base-v1/model.pt",
                                       "adapters": {"capsule-v1": "exports/textshape-lora-capsule-v1/adapter.pt"}}}
    for v in reg.values():
        v["checkpoint"] = str((REPO / v["checkpoint"]).resolve()) if not Path(v["checkpoint"]).is_absolute() \
            else v["checkpoint"]
        v["adapters"] = {k: str((REPO / p).resolve()) if not Path(p).is_absolute() else p
                         for k, p in v.get("adapters", {}).items()}
    return reg


AUTO_ADAPTER = "auto"


@lru_cache(maxsize=16)
def adapter_concepts(path: str) -> tuple[str, ...]:
    """The concept tokens an adapter was trained to add (its adapter_config.new_tokens)."""
    import torch
    return tuple(torch.load(path, map_location="cpu", weights_only=False)["adapter_config"]["new_tokens"])


def adapter_routing(model: str) -> dict[str, list[str]]:
    reg = model_registry().get(model, {"adapters": {}})
    return {name: list(adapter_concepts(p)) for name, p in reg["adapters"].items() if Path(p).exists()}


def route_adapter(model: str, prompt: str) -> tuple[str | None, str]:
    """adapter="auto": apply an adapter only when the prompt names its concept (text3d/routing.py, measured by E-T2)."""
    from ..text3d.routing import select_adapter
    return select_adapter(prompt, adapter_routing(model))


def check_model(req: TextTo3DRequest) -> None:
    reg = model_registry()
    if req.model not in reg or not Path(reg[req.model]["checkpoint"]).exists():
        raise ModelUnavailable(f"model {req.model!r} has no trained checkpoint on this server")
    if req.adapter and req.adapter != AUTO_ADAPTER and (req.adapter not in reg[req.model]["adapters"]
                        or not Path(reg[req.model]["adapters"][req.adapter]).exists()):
        raise ModelUnavailable(f"adapter {req.adapter!r} is not available for {req.model!r}")
    # Generator admission happens here, before any side effect: a request that would route to an unavailable
    # generator is refused with the reason instead of failing (or silently falling back) later.
    needed = {g for _, g in planned_generators(req)}
    for g in needed - {"berkelium-textshape"}:
        ok, why = _generator_available(g)
        if not ok:
            raise ModelUnavailable(f"generator {g!r} is unavailable: {why}")


def _generator_available(name: str) -> tuple[bool, str]:
    from ..text3d.generators import get_generator
    return get_generator(name).available()


def planned_generators(req: TextTo3DRequest) -> list[tuple[str, str]]:
    """[(prompt or part prompt, generator name)] this request will run. auto routes each prompt (or each planner
    part) with text3d.generators.route against the native checkpoint's vocabulary. No model call beyond the vocab."""
    from ..text3d import planner
    from ..text3d.generators import AUTO, route
    prompts = [req.prompt]
    if req.compose:
        plan, _ = planner.parse(req.prompt)
        if plan is not None:
            prompts = [plan.a, plan.b]
    if req.generator != AUTO:
        return [(p, req.generator) for p in prompts]
    vocab = native_vocab(req.model)
    out = []
    for p in prompts:
        g = route(p, vocab)["generator"]
        if g != "berkelium-textshape":           # first AVAILABLE generator in E-T3 evidence order (D-029)
            from ..text3d.generators import resolve_open_world
            g = resolve_open_world(_generator_available)[0] or g     # none available: check_model refuses with g
        out.append((p, g))
    return out


def native_vocab(model: str) -> list[str]:
    reg = model_registry()[model]
    return list(_load(reg["checkpoint"], None).vocab)


class SubprocessShapE:
    """Shap-E behind the Generator protocol, run by text3d.shap_e_worker in a child process: the API server never
    holds Shap-E's weights, and a memory-guard abort or watchdog kill fails the job, not the service. Stage progress
    is streamed back; cancellation is polled every 0.5 s and terminates the child. A finished child result in the
    job workdir is reused when the job resumes after a restart."""

    name = "shap-e-text300m"

    def __init__(self, steps: int, workdir: Path, threads: int = 2):
        from ..text3d.shap_e_backend import ShapEGenerator
        self.steps, self.workdir, self.threads = steps, Path(workdir), threads
        self._identity = ShapEGenerator(steps=steps, threads=threads).identity

    def available(self):
        from ..text3d.shap_e_backend import available
        return available()

    @property
    def identity(self) -> dict:
        return {**self._identity, "execution": "subprocess (berkelium_neural.text3d.shap_e_worker)"}

    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict:
        key = hashlib.sha256(json.dumps([prompt, seed, self.steps]).encode()).hexdigest()[:16]
        out = self.workdir / "shap_e" / key
        res = self._load(out)
        if res is not None:
            res["report"]["resumed_from_workdir"] = True
            return res
        from ..text3d.shap_e_worker import mem_available_mib
        free = mem_available_mib()
        need = int(os.environ.get("BERKELIUM_SHAP_E_MIN_FREE_MIB", SHAP_E_MIN_FREE_MIB))
        if free is not None and free < need:
            raise RuntimeError(f"RESOURCE_LIMIT: {free} MiB of host memory available, a Shap-E generation needs "
                               f"~{need} MiB (worker peak ~1.5 GB + watchdog margin); try again when memory is free")
        out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (str(PKG_ROOT), os.environ.get("PYTHONPATH"))
                                                           if p)}
        cmd = [sys.executable, "-m", "berkelium_neural.text3d.shap_e_worker", "--prompt", prompt, "--out", str(out),
               "--seed", str(seed), "--steps", str(self.steps), "--threads", str(self.threads)]
        with open(out / "worker.log", "w") as err:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, env=env)
            q: queue.Queue = queue.Queue()
            threading.Thread(target=_pump, args=(proc.stdout, q), daemon=True).start()
            try:
                while True:
                    if cancel and cancel():
                        raise Cancelled()
                    try:
                        line = q.get(timeout=0.5)
                    except queue.Empty:
                        if proc.poll() is not None and q.empty():
                            break
                        continue
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if ev.get("event") == "progress" and progress:
                        progress(ev["frac"], ev["stage"])
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        rc = proc.wait()
        res = self._load(out)
        if res is not None:
            return res
        tail = (out / "worker.log").read_text(errors="replace")[-1500:]
        if rc == 86:
            raise RuntimeError(f"RESOURCE_LIMIT: Shap-E worker stopped by its memory guard: {tail[-400:]}")
        if rc < 0:
            raise RuntimeError(f"RESOURCE_LIMIT: Shap-E worker killed by signal {-rc} (host memory watchdog?); "
                               f"{mem_available_mib()} MiB available now")
        raise RuntimeError(f"GENERATION_FAILED: Shap-E worker exited with {rc}: {tail}")

    @staticmethod
    def _load(out: Path):
        rp = out / "report.json"
        if not rp.exists():
            return None
        rep = json.loads(rp.read_text())
        if rep.get("status") == "ok" and (out / "mesh.npz").exists():
            z = np.load(out / "mesh.npz")
            return {"vertices": z["v"], "faces": z["f"], "colors": z["col"] if z["col"].size else None, "report": rep}
        if rep.get("status") == "invalid_mesh":
            return {"vertices": None, "faces": None, "colors": None, "report": rep}
        return None


def execution_record() -> dict:
    """Where this job's generation ran: this service host, and its GPU when torch sees one ("local-cpu" otherwise).
    Benchmark units produced on Kaggle carry their own hardware record in E-T3; this is the live service."""
    import platform
    rec = {"host": platform.node(), "device": "local-cpu"}
    try:
        import torch
        if torch.cuda.is_available():
            rec.update({"device": "local-gpu", "gpu": torch.cuda.get_device_name(0)})
    except ImportError:
        pass
    return rec


def _pump(stream, q: queue.Queue):
    for line in stream:
        q.put(line)


def _generator_for(req: TextTo3DRequest, ctx: JobContext):
    """Generator objects for this request: native in-process (service registry checkpoint/adapters), Shap-E in a
    subprocess; auto routes per prompt/part exactly as planned_generators does."""
    from ..text3d.generators import NATIVE, SHAP_E, NativeTextShape, get_generator
    reg = model_registry()[req.model]
    native = NativeTextShape(reg["checkpoint"], reg["adapters"], adapter=req.adapter, resolution=req.resolution)
    shap = SubprocessShapE(req.shap_e_steps, ctx.workdir)
    plan = dict(planned_generators(req))

    def pick(prompt: str):
        g = plan.get(prompt, req.generator)
        return native if g == NATIVE else shap if g == SHAP_E else get_generator(g)
    return pick


@lru_cache(maxsize=4)
def _load(ckpt: str, adapter: str | None):
    from ..text3d.train import load_model
    return load_model(Path(ckpt), Path(adapter) if adapter else None)


def _png(img: np.ndarray) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(b, format="PNG")
    return b.getvalue()


def make_text_to_3d(store: ArtifactStore):
    def run(request: dict, ctx: JobContext):
        from ..camera import orbit_cameras
        from ..text3d.generate import generate_mesh, mesh_to_gaussians, mesh_to_glb, mesh_to_ply
        from ..text3d.geometry import render_mesh
        req = TextTo3DRequest.model_validate(request)
        check_model(req)
        reg = model_registry()[req.model]
        adapter, routing = req.adapter, None
        if adapter == AUTO_ADAPTER:
            adapter, reason = route_adapter(req.model, req.prompt)
            routing = {"requested": AUTO_ADAPTER, "selected": adapter, "reason": reason}
        # Resumable: the mesh stage is checkpointed in the job workdir, so a job resumed after a crash reuses the
        # exact mesh (and artifact) it already published instead of publishing a second one.
        stage = ctx.workdir / "mesh_stage.npz"
        mesh = _resume_mesh(stage, store)
        if mesh is not None:
            mesh, v, f, col = mesh
            producer = dict(mesh.producer)
            ctx.add_artifact(mesh.id)
            ctx.progress(0.5, "mesh_from_checkpoint")
        elif _needs_open_world_path(req):
            mesh, v, f, col, producer = _open_world_mesh(req, ctx, store, stage)
        else:
            ctx.progress(0.05, "load_model")
            model = _load(reg["checkpoint"], reg["adapters"].get(adapter) if adapter else None)
            ctx.progress(0.15, "generate_sdf")
            out = generate_mesh(model, req.prompt, req.resolution, cancel=ctx.cancelled)
            ctx.check()
            rep = out["report"]
            if out["vertices"] is None or not rep["valid"]:
                raise RuntimeError(f"GENERATION_FAILED: no valid surface for this prompt ({rep.get('reason', rep)})")
            v, f, col = out["vertices"], out["faces"], out["colors"]
            producer = {"model": req.model, "adapter": adapter, **model.identity, "service": __version__,
                        "pipeline": "text->SDF->marching_cubes", **({"adapter_routing": routing} if routing else {}),
                        "generator": "berkelium-textshape", **_native_route_record(req)}
            cams = orbit_cameras(4, 3.2, 25.0, 128, 128)
            preview, _, _ = render_mesh(v, f, col, cams[0])
            ctx.progress(0.45, "write_mesh_artifact")
            mesh = store.create(ArtifactKind.TRIANGLE_MESH,
                                {"mesh.glb": (mesh_to_glb(v, f, col), "model/gltf-binary"),
                                 "mesh.ply": (mesh_to_ply(v, f, col), "application/x-ply"),
                                 "preview.png": (_png(preview), "image/png"),
                                 "report.json": (json.dumps(rep, indent=2, default=str).encode(),
                                                 "application/json")},
                                producer, config=req.model_dump(), validation=rep,
                                labels={"units": "normalized_object", "representation": "triangle_mesh",
                                        "not": "CAD model / measured geometry",
                                        "confidence": "model_output_unverified"})
            _save_mesh_stage(stage, mesh.id, v, f, col)     # before the job lists it: no window for a duplicate
            ctx.add_artifact(mesh.id)
        ctx.progress(0.55, "gaussian_init")
        scene = mesh_to_gaussians(v, f, col, n=4000, seed=req.seed)
        fit = {"steps": 0}
        if req.gaussian_fit_steps:
            scene, fit = fit_gaussians(scene, v, f, col, req.gaussian_fit_steps, ctx)
        ctx.progress(0.9, "write_gaussian_artifact")
        payload, man = scene.to_bytes()
        gauss = store.create(ArtifactKind.GAUSSIAN_SCENE,
                             {"scene.soa": (payload, "application/octet-stream"),
                              "scene.json": (json.dumps(man, indent=2).encode(), "application/json"),
                              "scene_3dgs.ply": (scene.to_3dgs_ply(), "application/x-ply")},
                             {**producer, "pipeline": "mesh_surface_samples->gaussians" +
                              ("+reference_renderer_fit" if fit["steps"] else "")}, parents=(mesh.id,),
                             config={"n": scene.n, "fit": fit}, validation={**scene.validate(), "fit": fit},
                             labels={"units": "normalized_object", "representation": "gaussian_scene",
                                     "not": "triangle mesh"})
        ctx.add_artifact(gauss.id)
        ctx.progress(0.93, "render_gaussians")
        img = render_scene_views(store, scene, gauss.id, producer)
        ctx.add_artifact(img.id)
    return run


def _needs_open_world_path(req: TextTo3DRequest) -> bool:
    """False when the request reduces to the native model on the whole prompt (generator=native, or auto routed
    to native, with no applicable planner): that path is the original executor code, unchanged."""
    from ..text3d import planner
    if req.compose and planner.parse(req.prompt)[0] is not None:
        return True
    return any(g != "berkelium-textshape" for _, g in planned_generators(req))


def _native_route_record(req: TextTo3DRequest) -> dict:
    out = {}
    if req.generator == "auto":
        from ..text3d.generators import route
        out["routing"] = [{"prompt": req.prompt, **route(req.prompt, native_vocab(req.model))}]
    if req.compose:
        from ..text3d import planner
        out.update({"planner": None, "planner_reason": planner.parse(req.prompt)[1]})
    return out


def _open_world_mesh(req: TextTo3DRequest, ctx: JobContext, store: ArtifactStore, stage: Path):
    """Mesh stage for generator != native or compose=True: one generator (auto routes it), or the two-object planner
    (each part generated and validated on its own, then placed). -> (mesh record, v, f, col, producer)."""
    from ..camera import orbit_cameras
    from ..text3d import planner
    from ..text3d.generate import mesh_to_glb, mesh_to_ply
    from ..text3d.generators import AUTO, route
    from ..text3d.geometry import render_mesh
    pick = _generator_for(req, ctx)

    def prog(lo, hi):
        return lambda fr, st: ctx.progress(lo + (hi - lo) * float(fr), st)
    plan, plan_reason = planner.parse(req.prompt) if req.compose else (None, "compose not requested")
    files, used = {}, {}
    try:
        if plan is not None:
            res, _ = planner.generate_scene(req.prompt, pick, req.seed, cancel=ctx.cancelled,
                                            progress=prog(0.05, 0.45))
            if res["status"] != "ok":
                raise RuntimeError(f"GENERATION_FAILED: planner part {res.get('failed_part')!r} produced no surface "
                                   f"({res.get('part_report', {}).get('reason')})")
            sc = res["scene"]
            v, f, col, rep = sc["vertices"], sc["faces"], sc["colors"], sc["report"]
            for p in res["parts"]:
                files[f"part_{p['role']}.glb"] = (mesh_to_glb(p["vertices"], p["faces"], p["colors"]),
                                                  "model/gltf-binary")
                used[p["generator"]] = p["generator_identity"]
            files["parts.json"] = (json.dumps({"plan": res["plan"], "parts": [
                {**{k: p[k] for k in ("prompt", "role", "generator", "transform", "report")},
                 "generation": {k: p["generation_report"].get(k) for k in
                                ("seconds", "sample_s", "decode_s", "valid", "n_faces", "memory_mib")}}
                for p in res["parts"]]}, indent=2, default=str).encode(), "application/json")
            pipeline = "planner(two parts)->per-part generation->bbox placement"
        else:
            g = pick(req.prompt)
            out = g.generate(req.prompt, req.seed, cancel=ctx.cancelled, progress=prog(0.05, 0.45))
            rep = out["report"]
            if out["vertices"] is None or not rep.get("valid"):
                raise RuntimeError(f"GENERATION_FAILED: no valid surface for this prompt ({rep.get('reason', rep)})")
            v, f, col = out["vertices"], out["faces"], out["colors"]
            used[g.name] = g.identity
            pipeline = PIPELINES.get("cadquery" if rep.get("backend") == "cadquery" else g.name, "text->SDF->marching_cubes")
            if g.name == "cadquery" or rep.get("backend") == "cadquery":   # script + exact STEP + core verdict
                files["script.py"] = ((rep.get("script") or "").encode(), "text/x-python")
                if out.get("step"):
                    files["result.step"] = (out["step"], "model/step")
                files["core_validation.json"] = (json.dumps({k: rep[k] for k in ("validation", "measurements",
                                                 "artifacts", "attempts", "scale_mm_per_unit")},
                                                 indent=2, default=str).encode(), "application/json")
            elif g.name == "cad":                        # the design itself and the core's verdict travel with the mesh
                files["design_proposal.json"] = (json.dumps(rep["proposal"], indent=2).encode(), "application/json")
                files["core_validation.json"] = (json.dumps({k: rep[k] for k in ("validation", "measurements",
                                                 "artifacts", "record_hash", "proposal_hash", "scale_mm_per_unit")},
                                                 indent=2, default=str).encode(), "application/json")
            for key, name in (("anchor", "anchor.png"), ("anchor_rgba", "anchor_rgba.png")):
                if out.get(key) is not None:            # zimage-triposr: the generated image is a first-class artifact
                    files[name] = (_png(out[key]), "image/png")
    except InterruptedError:
        ctx.check()
        raise
    ctx.check()
    open_world = bool(set(used) - {"berkelium-textshape", "cad", "cadquery"})
    decisions = None
    if req.generator == AUTO:
        vocab = native_vocab(req.model)
        from ..text3d.generators import resolve_open_world
        selected = dict(planned_generators(req))
        decisions = []
        for p in ([plan.a, plan.b] if plan else [req.prompt]):
            d = {"prompt": p, **route(p, vocab), "selected": selected.get(p)}
            if d["generator"] != "berkelium-textshape":
                d["skipped_unavailable"] = resolve_open_world(_generator_available)[1]
            decisions.append(d)
    first = next(iter(used.values()))
    producer = {"model": first.get("model"), "generator": req.generator if req.generator != AUTO else "+".join(used),
                "requested_generator": req.generator, "generator_identity": used, "service": __version__,
                "pipeline": pipeline, "seed": req.seed, "seconds": rep.get("seconds"), "execution": execution_record(),
                "peak_rss_mib": (rep.get("memory_mib") or {}).get("peak_VmRSS"),
                **({"routing": decisions} if decisions else {}),
                **({"planner": plan.as_dict()} if plan else {"planner": None, "planner_reason": plan_reason}
                   if req.compose else {})}
    cams = orbit_cameras(4, 3.2, 25.0, 128, 128)
    preview, _, _ = render_mesh(v, f, col if col is not None else np.full((len(v), 3), 0.7, np.float32), cams[0])
    ctx.progress(0.45, "write_mesh_artifact")
    labels = {"units": "normalized_object", "representation": "triangle_mesh", "not": "CAD model / measured geometry",
              "confidence": "model_output_unverified"}
    if open_world:
        labels["provenance"] = OPEN_WORLD_LABEL
    if set(used) in ({"cad"}, {"cadquery"}):           # core-realised geometry: millimetres before normalisation
        labels = {"units": f"normalized_object ({rep['scale_mm_per_unit']:.6g} mm per unit)",
                  "representation": "triangle_mesh (from Berkelium core geometry)",
                  "confidence": f"core validation {rep['validation']['summary']}",
                  "provenance": ("script proposed by the E-C2 CadQuery compiler; executed in the sandbox; geometry "
                                 "and measurements by OCCT through the Berkelium core") if "cadquery" in used or
                                rep.get("backend") == "cadquery" else
                                "design proposed by the E-C1 compiler; geometry and measurements by the Berkelium core"}
    mesh = store.create(ArtifactKind.TRIANGLE_MESH,
                        {"mesh.glb": (mesh_to_glb(v, f, col), "model/gltf-binary"),
                         "mesh.ply": (mesh_to_ply(v, f, col), "application/x-ply"),
                         "preview.png": (_png(preview), "image/png"),
                         "report.json": (json.dumps(rep, indent=2, default=str).encode(), "application/json"),
                         **files},
                        producer, config=req.model_dump(), validation=json.loads(json.dumps(rep, default=str)),
                        labels=labels)
    _save_mesh_stage(stage, mesh.id, v, f, col)
    ctx.add_artifact(mesh.id)
    return mesh, v, f, col, dict(mesh.producer)


def default_render_backend() -> str:
    """The GPU path (AMD gsplat forward + Berkelium deterministic HIP backward) when it is actually usable,
    otherwise the CPU reference renderer. Never inferred: gsplat_status() checks import, ROCm build and device."""
    from ..render.gsplat_adapter import gsplat_status
    return "berkelium_hip" if gsplat_status()["usable"] else "reference_cpu"


def render_scene_views(store: ArtifactStore, scene, parent_id: str, producer: dict, n_views: int = 4,
                       res: int = 256):
    """Splat-render the Gaussian scene itself (not the mesh) from an orbit of cameras -> one IMAGE artifact with
    a 2x2 contact sheet and the exact cameras used."""
    import torch

    from ..camera import orbit_cameras
    from ..render.gsplat_adapter import get_renderer
    from ..render.interface import GaussianTensors
    backend = default_render_backend()
    renderer = get_renderer(backend)
    cfg = RenderConfig(background=(1.0, 1.0, 1.0))
    dev = renderer.prepare(cfg)
    cams = orbit_cameras(n_views, 3.2, 25.0, res, res)
    g = GaussianTensors.from_scene(scene, device=dev)
    tiles, metas = [], []
    with torch.no_grad():
        for c in cams:
            o = renderer.render(g, c, cfg)
            tiles.append(o.rgb.detach().cpu().numpy())
            metas.append({k: v for k, v in o.meta.items() if isinstance(v, (int, float, str, bool))})
    cols = 2
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    sheet = np.concatenate(rows, 0)
    files = {"render.png": (_png(sheet), "image/png"),
             "cameras.json": (json.dumps([json.loads(c.model_dump_json()) for c in cams]).encode(),
                              "application/json")}
    for i, t in enumerate(tiles):
        files[f"view_{i}.png"] = (_png(t), "image/png")
    return store.create(ArtifactKind.IMAGE, files,
                        {"renderer": renderer.name, "backend": backend, "service": __version__,
                         "source_model": producer.get("model"), "pipeline": "gaussian_scene->splat_render"},
                        parents=(parent_id,), config={"views": n_views, "res": res, "background": cfg.background,
                                                      "profile": cfg.profile.value},
                        validation={"finite": bool(np.isfinite(sheet).all()), "mean": float(sheet.mean()),
                                    "std": float(sheet.std()), "meta": metas},
                        labels={"representation": "image", "not": "3D geometry",
                                "content": "splat render of the gaussian_scene parent"})


def _save_mesh_stage(path: Path, mesh_id: str, v, f, col):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, v=v, f=f, col=np.zeros((0, 3), np.float32) if col is None else col, mesh_id=np.array(mesh_id))
    os.replace(tmp, path)


def _resume_mesh(path: Path, store: ArtifactStore):
    """-> (mesh record, v, f, col) from this job's mesh checkpoint, or None if there is none (or its artifact is
    missing, in which case the mesh is regenerated)."""
    if not path.exists():
        return None
    ck = np.load(path, allow_pickle=False)
    mid = str(ck["mesh_id"])
    if not store.exists(mid):
        return None
    col = ck["col"] if ck["col"].size else None
    return store.get(mid), ck["v"], ck["f"], col


def fit_gaussians(scene, v, f, col, steps: int, ctx: JobContext):
    """Differentiable refinement of the Gaussian scene against z-buffer renders of the generated mesh
    (4 views, 64^2, white background) with the Berkelium reference renderer. Checkpoints every 25 steps."""
    import torch

    from ..camera import orbit_cameras
    from ..render.interface import GaussianTensors
    from ..render.reference import ReferenceRenderer
    from ..text3d.geometry import psnr, render_mesh
    cams = orbit_cameras(4, 3.2, 25.0, 64, 64)
    targets = [torch.from_numpy(render_mesh(v, f, col, c)[0]) for c in cams]
    g = GaussianTensors.from_scene(scene, requires_grad=True)
    opt = torch.optim.Adam([{"params": [g.means], "lr": 1e-3}, {"params": [g.color_logits], "lr": 2e-2},
                            {"params": [g.opacity_logits, g.log_scales, g.quats], "lr": 5e-3}])
    ck = ctx.workdir / "gaussian_fit.pt"
    start = 0
    if ck.exists():
        st = torch.load(ck, weights_only=False)
        for p, q in zip(g.params(), st["params"], strict=True):
            p.data.copy_(q)
        opt.load_state_dict(st["opt"])
        start = st["step"]
        before = st.get("psnr_before")
    R, cfg = ReferenceRenderer(), RenderConfig(background=(1.0, 1.0, 1.0))

    def score():
        with torch.no_grad():
            return float(np.mean([psnr(R.render(g, c, cfg).rgb.numpy(), t.numpy()) for c, t in zip(cams, targets,
                                                                                                strict=True)]))
    if start == 0:
        before = score()
    for step in range(start, steps):
        ctx.check()
        opt.zero_grad()
        loss = sum((R.render(g, c, cfg).rgb - t).abs().mean() for c, t in zip(cams, targets, strict=True)) / len(cams)
        loss.backward()
        opt.step()
        if (step + 1) % 25 == 0 or step + 1 == steps:
            torch.save({"params": [p.detach().clone() for p in g.params()], "opt": opt.state_dict(),
                        "step": step + 1, "psnr_before": before}, ck)
            ctx.checkpointed("gaussian_fit", {"step": step + 1})
            ctx.progress(0.55 + 0.35 * (step + 1) / steps, f"gaussian_fit {step + 1}/{steps}")
    from ..scene import GaussianScene
    out = GaussianScene(*(p.detach().numpy() for p in g.params()), meta={**scene.meta, "fit_steps": steps})
    return out, {"steps": steps, "views": 4, "psnr_before": before, "psnr_after": score(),
                 "target": "z-buffer renders of the generated mesh (self-consistency, NOT geometric accuracy)"}


def make_render(store: ArtifactStore):
    def run(request: dict, ctx: JobContext):
        import torch

        from ..render.gsplat_adapter import get_renderer
        from ..render.interface import GaussianTensors
        from ..scene import GaussianScene
        req = RenderRequest.model_validate(request)
        rec = store.get(req.artifact_id)
        if rec.kind != ArtifactKind.GAUSSIAN_SCENE:
            raise ValueError(f"INVALID_ARGUMENT: artifact {rec.id} is {rec.kind.value}, not a gaussian_scene")
        payload, _ = store.read_file(rec.id, "scene.soa")
        man = json.loads(store.read_file(rec.id, "scene.json")[0])
        scene = GaussianScene.from_bytes(payload, man)
        renderer = get_renderer(req.backend)
        dev = renderer.prepare(req.config)           # raises DEVICE_UNAVAILABLE before any allocation
        ctx.progress(0.3, f"render[{renderer.name}]")
        with torch.no_grad():
            o = renderer.render(GaussianTensors.from_scene(scene, device=dev), req.camera, req.config)
        img = store.create(ArtifactKind.IMAGE, {"render.png": (_png(o.rgb.cpu().numpy()), "image/png")},
                           {"renderer": renderer.name, **{k: o.meta[k] for k in ("backend", "profile",
                                                                                  "deterministic") if k in o.meta},
                            "service": __version__}, parents=(rec.id,),
                           config=json.loads(req.model_dump_json()), validation={"meta": o.meta},
                           labels={"representation": "image", "not": "3D geometry"})
        ctx.add_artifact(img.id)
    return run


def default_executors(store: ArtifactStore) -> dict:
    return {"text_to_3d": make_text_to_3d(store), "render": make_render(store)}

