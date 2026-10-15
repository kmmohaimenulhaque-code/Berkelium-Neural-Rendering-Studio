"""Allowlisted job executors: text_to_3d and render. Each stage is idempotent and cancellable; the optional
Gaussian fitting stage checkpoints to the job workdir and resumes after a restart."""

from __future__ import annotations

import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import __version__
from ..contracts import ArtifactKind, RenderConfig, RenderRequest, TextTo3DRequest
from .artifacts import ArtifactStore
from .jobs import JobContext

REPO = Path(__file__).resolve().parents[4]


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
                        "pipeline": "text->SDF->marching_cubes", **({"adapter_routing": routing} if routing else {})}
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

