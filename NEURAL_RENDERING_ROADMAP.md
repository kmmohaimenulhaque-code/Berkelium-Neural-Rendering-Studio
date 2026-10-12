# Neural Rendering Roadmap

**Revision:** `NR-0.2` (2026-10-12)  
**Status:** dependency-aware plan; all work below is future work unless marked existing in the integration map.

## Order of work

```mermaid
flowchart TD
    A[Archive/source/license inventory] --> B[Typed contracts + provenance]
    B --> C[CPU camera and renderer fixtures]
    C --> D[Independent HIP implementation]
    D --> E[Read-only artifact preview]
    E --> F[Provider inventory and E-A0]
    F --> G[Optional Z-Image-Turbo smoke]
    G --> H[TripoSR control assets]
    H --> I[Gaussian fitting and held-out validation]
    I --> J[Worker/persistence mutation phase]
```

## Milestones and exit gates

| ID | Work | Depends on | Exit evidence | Spend |
| --- | --- | --- | --- | --- |
| M0 | Record target archive, engineering reference, ownership and licenses. | None | `NEURAL_DATA_PROVENANCE.md`, source SHAs, unresolved items list. | CPU only. |
| M1 | Expand shared schemas for project, camera, artifact, job, error and provenance manifests. | M0 | Strict type/range/unknown-field tests; no filesystem or worker side effects on invalid input. | CPU only. |
| M2 | Implement CPU Gaussian forward/backward reference and camera adapters. | M1 | Renderer G1 fixtures, FP64 finite differences, serialization round trips, deterministic hashes. | CPU only. |
| M3 | Implement independent HIP backend and isolated worker. | M2, provider inventory | Renderer G2, 10-repeat determinism, bounded cancellation, measured memory/timing. | No allocation until M2; then E-A0 max 1.5 billed GPU-hours. |
| M4 | Install the read-only artifact preview path in the desktop boundary. | M1, security review | Preload isolation, no direct renderer HTTP/filesystem access, image/mesh/Gaussian metadata preview. | CPU/dev only. |
| M5 | Run optional image-prior smoke experiment. | M3 pass, model/license/operator preflight | E1 receipt with exact snapshot/config, finite images, measured limits, `geometry_generated=false`. | Separate approved tranche; do not assume T4/MI300X fit. |
| M6 | Run TripoSR controls and explicit mesh-to-Gaussian initialization. | M5 or owned control images, CPU extraction path | 3 control objects, valid mesh/bounds/units, 24-view orbit, rejection/failure examples. | CPU/T4 first; AMD only after approval. |
| M7 | Fit fixed-topology Gaussians and evaluate held-out views. | M2, M6 | G4 budgets, held-out metrics, provenance chain, no claim of independent geometry truth. | Bounded research run. |
| M8 | Add durable SQLite/artifact repositories and worker mutation phase. | M4, M7, auth/authorization review | Restart/cancel/reconcile tests, atomic artifact publication, explicit policy transition out of Phase 1. | Not part of current Phase 1. |

## Immediate implementation queue

1. `[D]` Write CPU contract fixtures using the exact camera and parameter conventions in `GAUSSIAN_RENDERER_SPEC.md`.
2. `[D]` Add strict validation and provenance-manifest tests before any model adapter.
3. `[D]` Build a read-only preview using artifact IDs and bounded metadata; do not add job submission.
4. `[D]` Request/read provider inventory only when the operator authorizes it; verify billing unit, one-GPU allocation, visible device, ROCm/PyTorch tuple, quota and balance.
5. `[D]` If M2 and the inventory pass, execute only the bounded E-A0/E0 experiment. Stop on cost ambiguity, compile failure, numerical mismatch, memory overage, or device reset.

## Explicit non-goals for this revision

- No source-code implementation, dependency installation, model download, GPU execution, fine-tuning, SDS, densification, SH appearance, multi-GPU training, CAD conversion, or production promotion.
- No claim that upstream CUDA code, benchmark numbers, model cards, or license badges prove ROCm compatibility or commercial permission.
- No attempt to make generated images into independent geometry ground truth. Unknown scale and hallucinated backs remain labelled and reviewed.

## Risk review cadence

At every milestone, record: hypothesis, exact inputs/revisions, actual output hashes, measured resource receipt, failures, remaining uncertainty, and why the next milestone is more informative than CPU-only work. A milestone can end as `pass`, `fail`, `blocked`, or `cancelled`; only the receipt may promote it to the next gate.

See `NEURAL_RENDERING_DECISIONS.md` for durable choices, `COMPUTE_BUDGET_AND_EXPERIMENTS.md` for the cost ledger, and `REPOSITORY_INTEGRATION_MAP.md` for current-versus-proposed implementation boundaries.
