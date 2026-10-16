<div align="center">

# BERKELIUM STUDIO

### An AI-Native Environment for 3D Creation, Neural Rendering & Scientific Visualization

**From natural language to 3D. From pixels to representations. From ideas to engineered worlds.**

Building an intelligent 3D workspace that brings AI-assisted modeling, differentiable rendering, scientific visualization, and GPU-accelerated computation into a unified desktop environment.

**Proudly Indian. Built for the World.**

[![Phase](https://img.shields.io/badge/Development-Phase%202-00D4AA?style=flat-square)](#-development-roadmap)
[![Platform](https://img.shields.io/badge/Platform-Desktop-171717?style=flat-square)](#-architecture)
[![GPU](https://img.shields.io/badge/GPU-AMD%20ROCm%20%2F%20HIP-ED1C24?style=flat-square)](#-neural-rendering)
[![AI](https://img.shields.io/badge/AI-3D%20Generation-7C3AED?style=flat-square)](#-natural-language-to-3d)

</div>

---

## The Vision

**Berkelium Studio is being developed to make advanced 3D creation more accessible, intelligent, and computationally powerful.**

Traditional 3D workflows require artists, engineers, and researchers to navigate complex tools, manually construct geometry, configure rendering pipelines, and move between disconnected applications.

Berkelium Studio aims to change that.

Imagine describing an object in natural language, generating an initial 3D representation, inspecting it in a professional viewport, refining it with AI assistance, and rendering it through a custom neural-rendering pipeline—all within one integrated environment.

The goal is not another chatbot wrapped around a 3D viewport.

The goal is a **unified AI-native workspace where language, geometry, rendering, code, and scientific data become parts of the same creative process.**

### What we're building

- **Natural-language 3D generation** — Generate 3D representations from text descriptions.
- **Neural rendering** — Develop a differentiable Gaussian Splatting renderer for GPU-accelerated reconstruction and optimization.
- **AI-assisted development** — Integrate a code editor and project-aware AI copilot into the creative workflow.
- **Scientific visualization** — Build foundations for visualizing complex spatial and scientific data.
- **Professional 3D workflows** — Develop scene management, project organization, viewport interaction, and extensible architecture.
- **AMD GPU acceleration** — Explore GPU-native compute and training workflows using HIP/C++ and the ROCm ecosystem.

These capabilities represent the product direction, not a claim that every subsystem is already implemented.

---

## What works today (verified 2026-10-16)

| Capability | Status | Evidence |
|---|---|---|
| Prompt → 3D in the Studio scene | **works** — Scene → *Generate*: the released BerkeliumTextShape model runs on the backend, the mesh is imported as a normal scene object (move, rotate, undo), its Gaussian scene is splat-rendered by the Berkelium renderer, and *Render this view* re-renders it from the live viewport camera | `tests/e2e/test_studio_text_to_3d.py` (real UI + real backend + released checkpoint), `results/e2e/` |
| Durable jobs | **works** — jobs and artifacts persist (SQLite + content-addressed store), resume after a crash without duplicate results, and Studio's *Recent* list reopens them after a restart | `services/neural/tests/test_durable_jobs.py` (SIGKILL mid-job), E2E restart step |
| Adapter routing | **works** — `auto` applies the capsule LoRA adapter only to prompts that name a capsule; vs always-on adapter on 32 held-out base prompts Chamfer 0.132 → 0.090 (paired 95% CI excludes 0), capsule gains kept | `results/text3d/E-T2_routing_cpu.report.md`, `test_routing.py` |
| Deterministic differentiable renderer (AMD) | **verified on one MI300X** — gsplat forward + Berkelium deterministic HIP backward: bit-identical gradients, forward bitwise-equal to gsplat, backward 1.32–2.21× gsplat | `results/renderer/E-R1_analysis.md` |
| Gaussian fitting to real objects (M7) | **works on CPU** — 8,000 Gaussians fitted to 24 views of 3 Google Scanned Objects scans: held-out PSNR 31.7–34.7 dB (+10 to +12.5 dB over init); coarse geometry improves, fine geometry on thin shapes does not | `results/m7/SUMMARY.md` |
| TripoSR image → 3D prior (M6) | **runs on CPU** (CUDA marching cubes replaced): 3/3 owned controls recognisable, closer to exact ground truth than our text model, merges side-by-side parts | `results/m6/REVIEW.md` |
| Open-world text → 3D, measured (E-T3) | **complete benchmark** — 12 real-object scans (B1), 20 semantic prompts (B2), 12 procedural (B3), 8 two-object (B4), CPU + Kaggle T4. Shap-E: B1 F@0.05 0.669, B2 R@1 0.95; Z-Image→TripoSR v2: 0.603 / 0.85 but better composition (B4 part count 0.50 with planner vs 0.00); native: 0.566 / 0.05. `auto` uses Shap-E for open-world prompts (rule D-029, written before the results) | `results/text3d/E-T3_openworld_cpu.report.md` |
| Z-Image-Turbo → TripoSR (M5, D-006) | **works on a GPU** (Kaggle 2× T4, CUDA; not ROCm): hash-verified 30.6 GB weights, bf16 (fp16 gives NaN on T4), 768² anchors ~122 s, U²-Net segmentation, TripoSR mesh; 80/80 E-T3 units valid. Unavailable (with the reason) on a CPU host | `results/m5/E-A1_receipt.json`, `services/neural/berkelium_neural/text3d/zimage_triposr.py` |
| Federated GPU runs | **works** — `scripts/kaggle_gpu_runner.py` ships a hashed code bundle (and hashed data) to a Kaggle GPU kernel and re-hashes every returned file before admitting it; stale or failed runs are refused | `results/kaggle/*/*/receipt.json`, `test_federated.py` |

Generated geometry is labelled `normalized_object` / `model_output_unverified`: it is model output, not a CAD
model or measured geometry. The text-to-3D model is small (362k parameters, trained on CPU) and its shapes are blobby.

### Quick start (CPU, Linux/macOS)

```bash
bash scripts/setup_cpu.sh --install          # .venv-cpu with pinned deps + pinned gsplat/HIP-CPU sources
bash scripts/start_backend.sh                # backend + neural service on http://127.0.0.1:8000
cd apps/desktop && npm ci && npm run dev     # Studio in the browser at http://localhost:5173 (or: npm run studio)
```

In Studio: **Scene** workspace → **Generate** → type a prompt (e.g. *a red cone on top of a blue box*) → **Generate**.
Headless, through the same API: `python scripts/demo_text_to_3d.py "a green capsule next to a white sphere"`.

Open-world prompts (optional, CPU): `bash scripts/setup_openworld.sh` installs Shap-E + CLIP (MIT) into `.venv-cpu`
and fetches ~3.1 GB of sha256-pinned weights. With the generator on **auto**, Studio sends prompts the native
model can read to it, and everything else (e.g. *a wooden chair with four legs*) to Shap-E. That takes about 15
minutes per object on a 2-vCPU machine (16 steps), needs ~2 GB of free RAM, and shows its stage, progress and
provenance. **plan 2 objects** splits *A on top of / next to B* into two separately generated, editable parts.
One generation from the command line: `.venv-cpu/bin/python scripts/openworld_generate.py "a teapot" --steps 16`.

Tests: `python -m pytest services/neural/tests -q` (CPU), `python -m pytest tests/e2e -q` (needs
`npm ci` and `python -m playwright install chromium`), `cd apps/desktop && npm test`. GPU runs: `docs/GPU_RUN_PLAN.md`.

## Development Status

Berkelium Studio is currently in **Phase 2: Generative 3D & Neural Rendering**.

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Desktop foundation and read-only intelligence | Foundation phase |
| Phase 2 | Text-conditioned 3D generation and neural rendering | Current development phase |
| Phase 3 | Integrated AI-assisted modeling, editing, and engineering workflows | Planned |

Phase 1 establishes the application architecture, desktop security boundaries, project foundations, scene structures, editor services, and local model-provider interfaces.

Phase 2 builds on that foundation to develop the first generation and neural-rendering pipeline.

Phase 3 aims to bring these capabilities together into a more complete AI-native 3D development environment.

> **Development principle:** Distinguish implemented functionality from experimental research and planned capabilities. Architecture is a foundation, not proof of a working feature.

---

# Phase 1 — Desktop Foundation

Phase 1 establishes the structural foundations required for a secure, extensible desktop application.

### Core subsystems

**Secure desktop runtime**
- Electron application architecture.
- Main-process and renderer-process separation.
- Restricted preload APIs and IPC boundaries.
- Context isolation and sandboxing.
- Validated inter-process communication.

**Project infrastructure**
- Project creation and reopening.
- Project manifests and metadata.
- Persistent project state.
- File and path validation.

**Scene foundations**
- Basic scene data structures.
- Scene creation and management.
- Foundations for future OpenUSD integration.

**Editor infrastructure**
- File browsing and loading.
- Read-only source inspection.
- Editor-state foundations.
- Project source-code search.

**Local AI connectivity**
- Ollama provider adapter foundation.
- LM Studio provider adapter foundation.
- Model connection handling.
- Project-question and query interfaces.

**Design system**
- Centralized design tokens.
- Color and spacing foundations.
- Reusable UI primitives.

**Contracts and permissions**
- Typed IPC contracts.
- Request validation.
- Structured error handling.
- Foundations for capability-based permissions.

### Phase 1 security boundary

Phase 1 intentionally excludes AI-driven mutations.

The renderer must not receive unrestricted access to the filesystem, shell, or privileged Electron APIs.

The architecture separates:

- **Trusted execution:** Main process and authorized service layers.
- **Restricted interaction:** Renderer and UI components.
- **Validated communication:** Explicit IPC contracts and request validation.

File editing, CAD feature creation, neural-rendering training, and arbitrary command execution are outside the Phase 1 capability set.

The purpose is to establish a controlled foundation before introducing AI systems capable of generating or modifying complex 3D assets.

---

# Phase 2 — Generative 3D & Neural Rendering

## The Mission

**Turn natural-language descriptions and visual observations into inspectable 3D representations through an integrated AI and rendering pipeline.**

Phase 2 focuses on five connected engineering objectives:

1. Natural-language-to-3D generation.
2. A custom differentiable Gaussian Splatting renderer.
3. A licensed, reproducible 3D training-data pipeline.
4. GPU-accelerated model training and rendering.
5. Integration with the Berkelium Studio desktop viewport.

The central objective is to establish a functional pipeline that connects AI inference to an actual 3D result—not simply produce a text response or a static image.

## 1. Natural Language → 3D

Develop a text-conditioned generation pipeline that transforms user descriptions into 3D representations suitable for inspection and further processing.

### Target workflow

```text
Natural-language prompt
          │
          ▼
     Text encoder
          │
          ▼
  Semantic conditioning
          │
          ▼
  3D representation
     generation
          │
          ▼
 Representation decoding
          │
          ▼
    3D scene output
          │
          ▼
 Berkelium Studio viewport
```

### Research and implementation goals

- Evaluate text-conditioned 3D generation architectures.
- Establish a suitable text-conditioning strategy.
- Generate or predict a target 3D representation.
- Develop representation-specific decoding and validation.
- Support export into formats suitable for downstream 3D workflows.
- Evaluate geometric quality, semantic consistency, and inference latency.

### Representation strategy

Different representations solve different problems. Phase 2 should evaluate them independently rather than force every use case into a single representation.

| Representation | Primary purpose |
|---|---|
| Triangle meshes | Editable geometry and conventional 3D workflows |
| Point clouds | Spatial reconstruction and geometric observations |
| 3D Gaussian Splats | View synthesis and neural scene representation |
| Signed distance fields | Implicit geometry and surface reconstruction |
| Volumetric representations | Scientific and volumetric visualization |

**Initial implementation target:** Choose a narrow, testable representation-generation workflow and make it functional end to end before expanding into additional representations.

Mesh generation and Gaussian Splatting should remain distinct capabilities unless an explicitly designed conversion or joint-optimization pipeline connects them.

## 2. Neural Rendering — Build Our Own Renderer

Develop a differentiable 3D Gaussian Splatting renderer using **C++ and HIP**, targeting compatible AMD GPUs through the ROCm ecosystem.

The objective is to understand and control the rendering pipeline rather than rely entirely on a third-party renderer.

### Core rendering components

- Gaussian representation and parameter storage.
- Camera models and view transformations.
- Projection of 3D Gaussians into screen space.
- Visibility ordering and tile-based rasterization.
- Gaussian contribution evaluation and alpha compositing.
- Image reconstruction and numerical validation.
- Differentiable rendering and gradient computation.
- Parameter optimization for scene reconstruction.
- GPU memory management and performance profiling.

### Proposed renderer architecture

```text
3D Scene / Gaussian Parameters
              │
              ▼
       Camera Transform
              │
              ▼
      Gaussian Projection
              │
              ▼
    Visibility / Tile Binning
              │
              ▼
       HIP Rasterization
              │
              ▼
       Alpha Compositing
              │
              ▼
        Rendered Image
              │
              ▼
    Image Reconstruction Loss
              │
              ▼
      Gradient Computation
              │
              ▼
      Parameter Optimization
              │
              └──────────► Updated Scene
```

### AMD GPU implementation

The renderer should be designed around measurable GPU workloads.

**Implementation priorities:**

- HIP kernels for computationally intensive rendering operations.
- Memory-efficient Gaussian parameter storage.
- Efficient visibility processing and rasterization.
- Numerically stable compositing.
- GPU profiling and regression benchmarks.
- Gradient correctness tests against reference implementations.
- Compatibility testing against the selected ROCm and hardware configurations.

HIP provides a programming model for GPU computation, but it does not automatically guarantee compatibility or optimal performance on every AMD GPU. The exact toolchain and hardware support must be verified.

### Renderer milestones

**Milestone A — Forward rendering**

Render a controlled collection of Gaussians from known camera positions and compare results against a trusted reference.

**Milestone B — Differentiability**

Validate gradients using finite-difference checks and reference implementations.

**Milestone C — Scene optimization**

Optimize Gaussian parameters against known target images and camera poses.

**Milestone D — Performance**

Profile GPU execution, memory usage, image quality, and scaling with scene complexity.

**Milestone E — Desktop integration**

Expose the renderer through a defined service interface and display its output in the Studio viewport.

Each milestone should produce reproducible tests and measurable results.

## 3. Training Data — Build a Reproducible Pipeline

A useful 3D generation and reconstruction system requires more than a large collection of models.

It needs appropriately licensed assets, consistent representations, reliable camera metadata, and reproducible preprocessing.

### Proposed data sources

- Licensed 3D assets.
- Multiview image collections.
- Camera intrinsics and extrinsics.
- Text descriptions and semantic labels.
- Meshes and geometric representations.
- Scene metadata and reconstruction targets.
- Synthetic training scenes where appropriate.

### Dataset pipeline

```text
Licensed Source Assets
         │
         ▼
  License Verification
         │
         ▼
   Asset Validation
         │
         ▼
 Geometry / Image Processing
         │
         ▼
 Camera-Pose Verification
         │
         ▼
 Text / Scene Annotation
         │
         ▼
 Dataset Normalization
         │
         ▼
 Train / Validation / Test
         │
         ▼
 Reproducible Data Manifests
```

### Dataset requirements

Each training sample should contain the fields appropriate to its representation and training objective.

Potential metadata includes:

- Asset and scene identifiers.
- Source and license information.
- Image dimensions and image references.
- Camera intrinsics and extrinsics.
- Geometry references.
- Text descriptions.
- Coordinate-system conventions.
- Units and normalization parameters.
- Preprocessing versions.
- Dataset split assignments.

Assets should not be included merely because they can be downloaded. Their licenses must permit the intended use, including model training where applicable.

The data pipeline should also track duplicate assets, corrupted geometry, inconsistent camera poses, and train/test contamination.

**Goal:** Make every training experiment traceable to a defined dataset version and preprocessing configuration.

## 4. Model Architecture

The architecture should remain modular so that the text encoder, representation generator, renderer, and optimization strategy can evolve independently.

### Proposed high-level pipeline

```text
                  USER PROMPT
                       │
                       ▼
                 Text Encoder
                       │
                       ▼
              Semantic Conditioning
                       │
                       ▼
           3D Representation Generator
                       │
                       ▼
              Scene Representation
                       │
                       ▼
            Differentiable Renderer
                       │
                       ▼
                Rendered Views
                       │
                       ▼
             Reconstruction Loss
                       │
                       ▼
               Optimization Loop
                       │
                       ▼
              Optimized 3D Scene
                       │
                       ▼
             Studio Integration
```

### Important architectural distinction

The text-conditioned generator and differentiable renderer have related but different responsibilities.

The generator predicts an initial 3D representation from semantic input. The renderer converts a representation into an image from a specified camera. During training or reconstruction, gradients may flow through the renderer to optimize supported scene parameters.

A text prompt alone does not provide ground-truth geometry or multiview supervision. The training methodology must define how semantic conditioning, 3D supervision, reconstruction objectives, and optimization are combined.

The final training architecture should be selected after evaluating model quality, available training data, GPU memory requirements, and compute budget.

## 5. Training and GPU Compute

Phase 2 requires a training workflow that is reproducible, observable, and compatible with the target GPU environment.

### Engineering priorities

- Dataset validation before training.
- Explicit training and evaluation splits.
- Reproducible model configurations.
- Checkpointing and recovery.
- Training metrics and experiment logs.
- GPU memory and utilization monitoring.
- Validation of generated 3D representations.
- Independent evaluation of rendering quality.
- Versioned model artifacts.

### Evaluation metrics

| Area | Measurements |
|---|---|
| 3D generation | Geometry validity, semantic consistency, surface quality |
| Novel-view synthesis | PSNR, SSIM, LPIPS where appropriate |
| Reconstruction | Image-space error and geometric accuracy where ground truth exists |
| Renderer | Frame time, throughput, GPU memory usage |
| Training | Loss curves, convergence behavior, validation metrics |
| Integration | Inference latency, data-transfer overhead, failure rate |

No benchmark result should be reported until the corresponding experiment has actually been executed.

## 6. Backend and Desktop Integration

The generation and rendering pipeline should run behind a defined backend interface.

The initial integration target is **FastAPI for service orchestration and inference endpoints**, with a controlled communication path to the Electron desktop application.

### Proposed architecture

```text
┌─────────────────────────────────────────────┐
│            BERKELIUM STUDIO                 │
│                                             │
│  Project Explorer     3D Viewport           │
│  Scene Hierarchy      AI Copilot            │
│  Asset Browser        Editor                │
└──────────────────────┬──────────────────────┘
                       │
               Validated IPC / API
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             SERVICE LAYER                   │
│                                             │
│  Project Service       Inference Client     │
│  Scene Service         Job Management       │
│  Asset Validation      Result Validation    │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│          GENERATION BACKEND                 │
│                                             │
│  FastAPI Endpoints                           │
│       │                                     │
│       ├── Text Conditioning                 │
│       ├── 3D Generation                     │
│       ├── Scene Validation                  │
│       └── Rendering Jobs                    │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│          GPU COMPUTE LAYER                  │
│                                             │
│  Model Inference                            │
│  HIP / C++ Renderer                         │
│  Reconstruction and Optimization            │
│  GPU Memory Management                      │
└─────────────────────────────────────────────┘
```

This diagram describes the proposed architecture, not a claim that all services are already implemented.

### Integration requirements

- Typed and versioned API contracts.
- Asynchronous job execution for long-running inference.
- Progress and status reporting.
- Input validation and request limits.
- Explicit error states and cancellation.
- Scene and asset serialization.
- Controlled transfer of generated artifacts.
- Safe integration with the desktop security model.

Generated files should enter the project through validated service interfaces. AI-generated content must not gain unrestricted filesystem or shell access.

### First end-to-end objective

A user enters a text description in Berkelium Studio.

The application submits a generation request, reports its status, receives a generated 3D artifact, validates the result, and loads it into the viewport.

That workflow is the central integration milestone for Phase 2.

---

# Architecture Principles

Berkelium Studio should grow through modular subsystems with explicit interfaces.

### 1. Representation independence

The application must not assume that every 3D asset is a mesh. Scene infrastructure should accommodate different representations without forcing incompatible data into one format.

### 2. GPU-aware computation

Rendering and training workloads should be designed around measurable compute and memory constraints.

### 3. Secure AI integration

Models should interact with the project through explicit capabilities and validated interfaces, not unrestricted access to the desktop runtime.

### 4. Reproducible research

Datasets, configurations, checkpoints, evaluation results, and preprocessing pipelines should be versioned or traceable.

### 5. Incremental delivery

Every major subsystem should produce a working, testable milestone before additional complexity is introduced.

### 6. Honest engineering

Experimental components, mocked interfaces, research prototypes, and production-ready features must be clearly distinguished.

---

# Development Roadmap

## Phase 1 — Desktop Foundation

**Objective:** Establish a secure desktop application and read-only project intelligence.

Focus areas:
- Electron architecture and IPC boundaries.
- Project creation and persistence.
- Scene data foundations.
- Editor and source-inspection services.
- Local model-provider interfaces.
- Design tokens and UI primitives.
- Request validation and permissions.

## Phase 2 — Generative 3D & Neural Rendering

**Objective:** Connect text-conditioned 3D generation, a differentiable renderer, and the Studio viewport.

Focus areas:
- Evaluate and select a 3D generation architecture.
- Establish a licensed training-data pipeline.
- Implement forward Gaussian Splatting rendering in HIP/C++.
- Validate differentiability and reconstruction optimization.
- Develop reproducible training and evaluation workflows.
- Expose inference through FastAPI.
- Display validated generation results in the desktop viewport.

### Phase 2 exit criteria

Phase 2 should not be considered complete until:

- A defined text-to-3D pipeline produces valid, inspectable 3D output.
- The renderer produces validated output for controlled test scenes.
- Gradient correctness has been tested against an appropriate reference.
- A reconstruction or optimization experiment demonstrates the intended differentiable workflow.
- Training data and licensing information are traceable.
- The backend can process generation requests and return validated artifacts.
- The desktop application can display those artifacts in its viewport.
- Automated tests cover critical interfaces and failure cases.
- Performance and quality results are recorded using reproducible benchmarks.

## Phase 3 — AI-Native 3D Engineering

**Objective:** Evolve Berkelium Studio into an integrated environment for AI-assisted 3D creation and engineering.

Potential development areas:

- Project-aware AI copilot.
- Code editor and development tools.
- Controlled AI-assisted scene editing.
- Parametric geometry and CAD integration.
- Expanded neural-rendering workflows.
- Scientific visualization.
- Scene-aware tools and workflow automation.
- Capability-based permissions for approved mutations.
- More advanced asset generation and optimization.

These capabilities remain planned until implemented and validated.

---

# Repository Structure

The following is the intended high-level monorepo organization. Actual paths and package availability should be verified against the repository.

```text
berkelium-studio/
│
├── apps/
│   └── desktop/
│       └── src/
│           ├── main/
│           ├── preload/
│           └── renderer/
│
├── packages/
│   ├── contracts/
│   ├── design-system/
│   ├── project-core/
│   ├── scene-core/
│   ├── editor-core/
│   └── provider-core/
│
├── services/
│   ├── project-service/
│   └── generation-service/       # Planned
│
├── engine/
│   └── neural-renderer/           # Planned HIP/C++ implementation
│
├── ml/
│   ├── generation/                # Planned
│   ├── datasets/                  # Planned data tooling
│   └── training/                 # Planned
│
├── tests/
│
├── package.json
├── pnpm-workspace.yaml
└── README.md
```

New directories should be introduced as the corresponding implementation begins, rather than created merely to make the repository appear complete.

---

# Getting Started

See **Quick start** above for the verified commands. Details:

- Python environments are separate on purpose (`configs/environments/README.md`): `.venv-cpu` (CPU reference,
  backend, tests), the AMD ROCm image for gsplat + the Berkelium HIP kernel, `.venv-triposr` for M6, and
  `configs/environments/zimage-rocm.txt` for M5.
- `config.json` in the project root is optional (chat providers, backend API key, allowed origins); the neural
  service needs no credentials. Neural jobs/artifacts live in `$BERKELIUM_NEURAL_HOME` (default `~/.berkelium/neural`).
- The desktop app: `cd apps/desktop && npm ci && npm run studio` (Electron + Vite), or `npm run dev` for the browser.
- Reproducing results: `scripts/run_benchmarks.py --experiment configs/experiments/<id>.json` (renderer, text-to-3D),
  `scripts/fit_real_object.py` (M7), `.venv-triposr/bin/python scripts/m6_triposr_controls.py` (M6),
  `scripts/m5_image_prior_smoke.py --check` (M5). `docs/ENGINEERING_LOG.md` records what ran, where, and the outcome.

---

# Contributing

Berkelium Studio is being developed as a modular engineering project spanning desktop infrastructure, AI, computer graphics, and GPU compute.

Areas of technical interest include:

- 3D generation and reconstruction.
- Gaussian Splatting and differentiable rendering.
- AMD HIP/C++ GPU programming.
- 3D asset processing and dataset tooling.
- Electron desktop architecture.
- Scientific visualization.
- AI-assisted development environments.

Before submitting code or distributing modified versions, review the repository's applicable license and contribution requirements.

**Source availability does not automatically grant redistribution or commercial-use rights.** Contributions must also comply with the project's contribution terms.

---

# License

Berkelium Studio is intended to use the Berkelium Source-Available License, subject to the actual license file committed to this repository.

The license governs the permissions applicable to the software. Commercial use and redistribution require authorization where stipulated by the applicable license.

Third-party dependencies remain subject to their respective licenses.

---

<div align="center">

## Building the Next Generation of 3D Software

Berkelium Studio brings together AI, geometry, rendering, and engineering in pursuit of a more intelligent creative workflow.

**Proudly Indian. Built for the World.**

</div>
