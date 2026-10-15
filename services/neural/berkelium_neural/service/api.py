"""FastAPI router for neural jobs and artifacts (mounted by services/backend/main.py under /v1/neural).

Every request is validated (pydantic contracts, id regexes, model availability, queue admission) BEFORE any
filesystem, model or GPU side effect. Standalone: ``uvicorn berkelium_neural.service.api:app``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import Field, TypeAdapter, ValidationError

from ..contracts import RenderRequest, TextTo3DRequest
from .artifacts import ArtifactError, ArtifactStore
from .executors import ModelUnavailable, check_model, default_executors, model_registry
from .jobs import AdmissionError, JobManager

JobBody = Annotated[TextTo3DRequest | RenderRequest, Field(discriminator="kind")]
_job_adapter = TypeAdapter(JobBody)
_ERR_STATUS = {"NOT_FOUND": 404, "INVALID_ARGUMENT": 422, "CORRUPTED": 500, "INVALID_PARENT": 422}


class NeuralService:
    def __init__(self, root: Path | None = None, max_pending: int = 8):
        self.root = Path(root or os.environ.get("BERKELIUM_NEURAL_HOME", Path.home() / ".berkelium" / "neural"))
        self.store = ArtifactStore(self.root / "artifacts")
        self.jobs = JobManager(self.root / "jobs", default_executors(self.store), max_pending=max_pending)

    def close(self):
        self.jobs.shutdown()


def create_router(svc: NeuralService) -> APIRouter:
    r = APIRouter(prefix="/v1/neural", tags=["neural"])

    @r.get("/capabilities")
    def capabilities():
        from ..render.gsplat_adapter import gsplat_status
        reg = model_registry()
        models = {k: {"checkpoint_present": Path(v["checkpoint"]).exists(),
                      "adapters": {a: Path(p).exists() for a, p in v["adapters"].items()}} for k, v in reg.items()}
        gs = gsplat_status()
        from ..text3d.generators import generator_status
        from .executors import adapter_routing
        gens = generator_status()             # availability reasons + identities; loads no weights
        native_ok = bool(models.get("berkelium-textshape", {}).get("checkpoint_present"))
        for k in ("berkelium-textshape", "auto"):
            gens[k]["available"] = native_ok
            gens[k]["reason"] = "" if native_ok else "native checkpoint missing on this server"
        return {"job_kinds": sorted(svc.jobs.executors), "models": models, "generators": gens,
                "adapter_routing": {k: adapter_routing(k) for k in reg},
                "renderers": {"reference_cpu": {"available": True},
                              "gsplat": {"available": gs["usable"], "pin": gs["pin"]["commit"],
                                         "reason": None if gs["usable"] else "requires ROCm PyTorch + GPU + gsplat"},
                              "berkelium_hip": {"available": True, "execution": "gpu" if gs["torch_hip"] and
                                                gs["device_visible"] else "cpu_emulated_hip (HIP-CPU)"}},
                "limits": {"max_pending_jobs": svc.jobs.max_pending, "max_prompt_chars": 300}}

    @r.post("/jobs", status_code=202)
    def submit(body: Annotated[dict, Body()]):
        try:
            req = _job_adapter.validate_python(body)
        except ValidationError as e:
            raise HTTPException(422, detail=e.errors(include_url=False, include_context=False)) from None
        try:
            if isinstance(req, TextTo3DRequest):
                check_model(req)
            else:
                svc.store.get(req.artifact_id)
            return svc.jobs.submit(req.model_dump(mode="json"))
        except ModelUnavailable as e:
            raise HTTPException(503, detail={"code": e.code, "message": str(e)}) from None
        except ArtifactError as e:
            raise HTTPException(_ERR_STATUS.get(e.code, 400), detail={"code": e.code, "message": str(e)}) from None
        except AdmissionError as e:
            raise HTTPException(429 if e.code == "RESOURCE_LIMIT" else 422,
                                detail={"code": e.code, "message": str(e)}) from None

    @r.get("/jobs")
    def list_jobs(limit: int = 20):
        return svc.jobs.list(max(1, min(limit, 200)))

    def _job(job_id: str):
        if len(job_id) != 32 or not all(c in "0123456789abcdef" for c in job_id):
            raise HTTPException(422, detail={"code": "INVALID_ARGUMENT", "message": "malformed job id"})
        j = svc.jobs.get(job_id)
        if j is None:
            raise HTTPException(404, detail={"code": "NOT_FOUND", "message": "job not found"})
        return j

    @r.get("/jobs/{job_id}")
    def get_job(job_id: str):
        j = _job(job_id)
        # reconcile on read: a listed artifact whose record is gone is reported, never silently served
        missing = [a for a in j.artifacts if not svc.store.exists(a)]
        return {**j.model_dump(mode="json"), "transitions": svc.jobs.transitions(job_id), "missing_artifacts": missing}

    @r.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        _job(job_id)
        return svc.jobs.cancel(job_id)

    @r.get("/artifacts")
    def list_artifacts(limit: int = 20):
        return svc.store.list(max(1, min(limit, 200)))

    @r.get("/artifacts/{aid}")
    def get_artifact(aid: str):
        try:
            return svc.store.get(aid)
        except ArtifactError as e:
            raise HTTPException(_ERR_STATUS.get(e.code, 400), detail={"code": e.code, "message": str(e)}) from None

    @r.get("/artifacts/{aid}/files/{name}")
    def get_file(aid: str, name: str):
        try:
            data, f = svc.store.read_file(aid, name)
        except ArtifactError as e:
            raise HTTPException(_ERR_STATUS.get(e.code, 400), detail={"code": e.code, "message": str(e)}) from None
        return Response(data, media_type=f.media_type, headers={"ETag": f'"{f.sha256}"',
                                                                "Cache-Control": "public, max-age=31536000, immutable"})

    return r


def create_app(svc: NeuralService | None = None) -> FastAPI:
    from contextlib import asynccontextmanager

    from fastapi.middleware.cors import CORSMiddleware
    service = svc or NeuralService()

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(title="Berkelium Neural Service", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("BERKELIUM_ALLOWED_ORIGINS",
                       "http://localhost:5173,http://127.0.0.1:5173").split(","), allow_methods=["GET", "POST"],
                       allow_headers=["*"])
    app.include_router(create_router(service))
    app.state.neural = service

    @app.get("/health")
    def health():
        return {"status": "ok", "components": {"neural": {"status": "ok"}}}
    return app


def __getattr__(name):     # lazy module-level ``app`` for ``uvicorn berkelium_neural.service.api:app``
    if name == "app":
        return create_app()
    raise AttributeError(name)
