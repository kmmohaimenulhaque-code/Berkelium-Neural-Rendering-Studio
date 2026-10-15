# M6 review — TripoSR control assets (2026-10-15)

Receipt: `M6_receipt.json` (all numbers). Inputs: three Berkelium-owned control images (`control_*.png`), shaded
renders of frozen test-split assets of `berkelium-procedural-shapes-v1` (MIT), so ground truth is exact and was seen
by neither model in training. Run on CPU (2 threads) with the D-020 compatibility branch; 47–56 s per input after a ~16 s load.

## Stage C gates (TEXT_TO_3D_RESEARCH_PLAN.md §5)

| Gate | Result |
|---|---|
| finite mesh, non-empty faces | 3/3 |
| Berkelium mesh validator | 3/3 after `clean_mesh` (raw: 1/3 — marching cubes left one zero-area face on two meshes) |
| bounds/units metadata | recorded (TripoSR canonical frame, radius 0.87; not metric) |
| renderable 24-view orbit | 3/3 (`triposr_*_orbit24.png`) |
| ≤120 s/input after load | 3/3 on CPU (see `infer_extract_s_cpu`; the gate was written for a GPU) |
| host extraction scratch ≤ 8 GiB | peak process RSS 2.6–2.9 GiB, model included (memory-mapped weights) |
| peak ≤ 16 GiB GPU | not applicable (CPU run) |
| ≥ 2/3 controls recognisable, no major duplicated-front geometry | **3/3 recognisable** — visual review below |

## Visual review (reviewer: Claude, AI assistant; not a human review)

- `on_top_of__cylinder_sphere__000`: green cylinder on a cyan sphere, correct proportions and colours; tilted by
  ~16° because the control camera has 16° elevation and TripoSR assumes a level camera (its pose prior). No
  duplicated front.
- `next_to__box_box__000`: blue box and smaller yellow box, correct colours and relative size; the two boxes are
  **merged into one component** (ground truth: 2 separate parts).
- `next_to__torus_torus__000`: cyan and purple tori with visible holes; **merged into one component** where they
  approach (ground truth: 2).

Part count is therefore correct for 1/3 (the single-component stack) and wrong for both "next to" pairs.

## Comparison (Chamfer-L1 / F-score vs exact GT, same similarity alignment for every candidate)

TripoSR (input: the control image) is closer to ground truth than Berkelium's text-to-3D model (input: the asset's
caption) on all three controls; the gap is small on the torus pair. These candidates receive different inputs (image
vs text), so this compares the two priors available to the pipeline, not two models on one task. The Berkelium model
in its native frame (no alignment) scores within ~13% of its aligned score (better on the torus pair), i.e. the
alignment does not hide a frame error.

## Consumed by the pipeline

Each TripoSR mesh is converted to 4,000 surface Gaussians and splat-rendered by the Berkelium reference renderer
(`triposr_*_gaussians.png`): the D-006 path (image → TripoSR mesh prior → Gaussians) runs end to end on these
controls. Not done: TripoSR on generated (Z-Image) anchors — that needs M5.
