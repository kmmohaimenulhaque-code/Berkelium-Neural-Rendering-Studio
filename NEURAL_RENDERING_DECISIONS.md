# Neural Rendering Decisions

**Revision:** `NR-0.2` (2026-10-12)  
**Status:** proposed architecture decisions; no implementation or execution is claimed.

Evidence labels: `[O]` inspected implementation observation, `[F]` external source fact, `[D]` Berkelium decision, `[H]` hypothesis requiring an experiment.

## Decision register

| ID | Decision | Rationale | Revisit trigger |
| --- | --- | --- | --- |
| D-001 | `[D]` Keep the desktop renderer untrusted; use a narrow preload/main bridge, loopback FastAPI, and a trusted worker. | `[O]` The current Electron window has Node integration enabled and no preload; direct renderer HTTP also exposes the backend boundary. | A measured alternative provides equal isolation and auditable authorization. |
| D-002 | `[D]` Keep Phase 1 read-only. No training, scene mutation, arbitrary commands, or job submission is enabled by documentation. | `[O]` README Phase 1 excludes mutation and neural training; the current API has no neural routes. | A later approved phase passes auth, persistence, worker-isolation, and migration gates. |
| D-003 | `[D]` Berkelium owns an independent CPU reference and HIP/C++ renderer contract. | Avoid importing Graphdeco's restricted implementation; keep CPU semantics and HIP behavior testable without model weights. | License review authorizes a compatible backend and parity testing justifies an adapter. |
| D-004 | `[D]` Canonical scene storage is GaussianScene v1 metadata plus checksummed binary arrays; meshes, images, CAD, checkpoints, and reports stay distinct artifact kinds. | Prevents a `.ply`, mesh, or preview from being misrepresented as a CAD or measured scene. | A versioned interchange requirement proves a different representation is necessary. |
| D-005 | `[D]` Use right-handed +Y-up world coordinates and OpenCV +Z-forward camera coordinates with explicit adapters. | Avoids silent Blender/OpenGL/COLMAP/MVDream basis and pose inversions. | A concrete external format requires a new canonical basis and all fixtures are migrated. |
| D-006 | `[D]` First text-to-3D path is Z-Image-Turbo → human-reviewed image → TripoSR explicit 3D prior → optional Gaussian fitting. | It avoids making independent view prompts appear calibrated and avoids making Zero123++/InstantMesh's non-commercial weights a default dependency. | A licensed direct or joint-multiview system wins the same control geometry/consistency gates. |
| D-007 | `[D]` Treat Mip-NeRF 360 as a real-capture/unbounded-scene comparison, not text-to-3D supervision. | Its posed-capture objective and scene normalization differ from prompt generation. | A matched permitted dataset and protocol establish a valid cross-representation comparison. |
| D-008 | `[D]` CPU analytic/finite-difference fixtures precede any MI300X allocation; first allocation is one GPU and at most 1.5 billed GPU-hours. | Reduces paid debugging and catches camera/math errors without hardware. | Provider billing or allocation inventory changes the approved envelope. |
| D-009 | `[D]` Protect at least 8 GPU-hours and never infer cost from wall time alone. | Provider billing unit, GPU count, rounding, and balance are not verified. | A provider receipt establishes a different accounting model. |
| D-010 | `[D]` Use canonical job states `pending`, `running`, `completed`, `failed`, `cancelled`, `checkpointed`, and `resumed`; keep experiment outcomes in receipts. | Separates lifecycle state from research results such as `blocked` or `correctness_pass/performance_stop`. | A versioned API migration defines a replacement state machine. |

## Rejected or deferred choices

- `[D]` Do not use the original Graphdeco implementation as Berkelium source; its license is restrictive for the intended product posture.
- `[D]` Do not use Zero123++ v1.2/InstantMesh as the default production path; Zero123++ weights are CC-BY-NC-4.0 and every fine-tuned/base component requires separate review.
- `[D]` Do not claim AMD support from a CUDA wheel, a successful import, or the presence of `cuda:0` in PyTorch HIP.
- `[D]` Do not call a mesh, NeRF, or Gaussian preview a CAD model or measured geometry without a separate accuracy and units contract.
- `[H]` Do not assume TripoSR's CUDA extraction extension works on ROCm; CPU extraction or a verified replacement is the first compatibility branch.

## Consequences

These decisions intentionally delay polished end-to-end generation. They make the first measurable result a reproducible renderer contract and artifact lineage, not an attractive but unverified image. Detailed interfaces are in `NEURAL_RENDERING_API.md`, numeric semantics in `GAUSSIAN_RENDERER_SPEC.md`, and execution ordering in `NEURAL_RENDERING_ROADMAP.md`.
