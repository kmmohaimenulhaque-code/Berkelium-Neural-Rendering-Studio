# Compute Budget and Experiments

**Status:** architecture and experiment-control plan; no GPU work authorized by this document.

## Executive rule

The reported MI300X balance is approximately **27 GPU-hours**, but it is not yet a verified balance. No MI300X execution is permitted until the provider confirms the billing unit, allocated GPU count, visible device count, quota, and remaining balance. A one-hour run on a four-GPU allocation may consume four GPU-hours or more depending on the provider's accounting. The default first tranche is **1–2 billed GPU-hours**, one GPU only, followed by a go/no-go review.

## Confirmed, reported, and unknown resources

| Resource | Current evidence | Planning meaning |
|---|---|---|
| AMD MI300X | User reports about 27 remaining free hours; earlier diagnostics reported Ubuntu 24.04, ROCm 7.0, PyTorch `2.10.0+rocm7.0`, HIP `7.0.51831`, `gfx942`, and about 196 GB visible memory. | Treat all values as historical until a read-only inventory confirms them. Use one GPU by default. |
| Kaggle T4 | User reports access to T4-class notebooks. | CUDA/PyTorch reference work, preprocessing, small inference, and evaluation only; verify actual allocation and session limits each run. |
| Colab T4 | User reports access to T4 notebooks. | Independent compatibility and inference checks; never assume availability or durable runtime. |
| Hugging Face | Hub/model/dataset access is available or intended. | Distribution, pinned revisions, metadata, and artifacts; hosted GPU is optional and must be cost/quota-verified. |
| Target repository | Local extracted Studio source is present, but this workspace has no `.git` metadata; GitHub target returned 404 to unauthenticated inspection. | Do not claim branch/working-tree state for the remote target. Preserve local files and request authenticated access before remote changes. |
| Engineering source | Public GitHub repository is reachable; API metadata reports default branch `main`, public visibility, and no declared repository license. | Reference/integration source only. No code transfer until file-level licensing and ownership are cleared. |

## Required inventory before any GPU allocation

Record command output in an experiment manifest without secrets:

```text
provider account/quota:          <provider-specific read-only check>
billing unit:                    <GPU-hour, instance-hour, credit, unknown>
allocated GPU count:             <integer>
visible GPU count:               <integer>
GPU model / architecture:        <model> / <gfx arch>
VRAM per visible device:         <bytes>
ROCm / HIP / PyTorch versions:   <versions>
compiler and profiler:           <paths and versions>
device affinity:                 <HIP_VISIBLE_DEVICES / container mapping>
disk free / durable output:      <bytes / destination>
```

The inventory is a gate, not a benchmark. It must not install packages, download weights, or start training.

## Allocation policy

1. CPU first: contracts, camera validation, dataset manifests, reference math, serialization, API tests, and synthetic fixtures.
2. Kaggle/Colab T4 next: CUDA-compatible reference implementations, low-resolution inference, metric scripts, and dependency experiments.
3. MI300X only for actual HIP correctness, ROCm-only compatibility, large-memory tests, or a measured renderer optimization.
4. Never request multiple MI300X devices unless a single-device experiment fails for a documented reason and the extra allocation has an explicit cost/benefit calculation.
5. Keep **at least 8 GPU-hours unallocated** until custom renderer correctness and Studio integration show measurable progress. The approximate 27-hour planning envelope is therefore not fully spendable.

## First high-information experiment

**Experiment E0 — single-scene, low-resolution renderer and ROCm feasibility gate.**

Before MI300X use, run the CPU reference and a T4-compatible baseline on a small crop/subset of one licensed Mip-NeRF 360 scene. Do not train a text-to-3D model, fine-tune Z-Image, or process a full scene collection.

| Item | Budget and method |
|---|---|
| MI300X allocation | 1 GPU, maximum 1.5 billed GPU-hours; hard timeout at 90 minutes wall time unless provider billing is not GPU-hour based, in which case stop and recalculate. |
| Workload | HIP build/smoke test, 256–512 px forward render, 1–4k Gaussians, finite-value checks, and one short optimization loop against CPU/reference output. |
| Expected outputs | Environment manifest, kernel build log, rendered PNG/NPZ, loss curve, forward/backward timings, peak VRAM, Gaussian count, checksum, and failure classification. |
| Acceptance | Compiles without source patching in the runtime; deterministic output for fixed seed; no NaN/Inf; forward mixed error no worse than the predeclared tolerance; backward finite-difference sign agreement on a small fixture; no unexplained device reset. |
| Stop condition | Stop immediately if the HIP extension cannot compile, the reference comparison diverges beyond tolerance, memory exceeds the measured envelope, or the provider reports unexpected multi-GPU billing. A failed hypothesis is useful; do not extend the run to obtain a better-looking image. |
| Go condition | Proceed only if all acceptance checks pass and the actual billed cost is within the approved tranche. Reassess with observed timings rather than authorizing a full run automatically. |

The first experiment answers the highest-risk question: whether the custom HIP path can execute a small, verifiable differentiable renderer on the actual AMD allocation. It does not claim reconstruction quality or text-to-3D capability.

## Cumulative budget ledger

| Experiment | Resource | Max billed GPU-hours | Actual GPU-hours | Gate |
|---|---|---:|---:|---|
| E0 HIP feasibility and gradient smoke test | 1× MI300X | 1.5 | `TBD` | Architecture approval + provider inventory |
| E1 T4 reference reconstruction and metrics | Kaggle or Colab T4 | 0 MI300X | `TBD` | E0 or CPU reference available; use notebook quota only |
| E2 HIP renderer scale test | 1× MI300X | 2.0 | `TBD` | E0 pass; target one bottleneck only |
| E3 small end-to-end scene optimization | 1× MI300X | 3.0 | `TBD` | E2 pass and quality baseline recorded |
| E4 integration/debug reserve | 1× MI300X | 4.0 | `TBD` | Only for a named blocker with reproducible failure |
| E5 final validation reserve | 1× MI300X | 4.0 | `TBD` | Only after API/artifact/renderer integration is testable |
| **Protected reserve** | 1× MI300X | **8.0 minimum** | `TBD` | Must not be allocated in advance |
| **Uncommitted remainder** | Provider-dependent | **4.5+** | `TBD` | Recalculate after billing verification |

The ledger must be updated after every allocation from the provider receipt, not from wall-clock guesses. If the provider bills per instance rather than GPU, record both wall time and billed credit units.

## Resource placement matrix

| Work | CPU | Kaggle/Colab T4 | MI300X/HIP | Hugging Face |
|---|---:|---:|---:|---:|
| Contracts, schemas, API tests | Primary | No | No | No |
| Camera validation and manifests | Primary | Optional | No | Dataset metadata only |
| CPU renderer/reference gradients | Primary | Optional | No | No |
| CUDA reference renderer | No | Primary | No | No |
| HIP kernel correctness | No | Cannot validate | Required | No |
| MI300X performance tuning | No | Cannot validate | Required, late | No |
| Z-Image smoke test | Optional | First choice | Only if T4 fails and budget is approved | Model revision source |
| Full-resolution/multi-scene work | No by default | Not authorized by default | Not authorized by default | Artifact distribution only |

CUDA execution cannot validate HIP device correctness, and CPU agreement cannot establish AMD performance.

## Memory and storage estimates

These are planning estimates, not measurements. Measure them on the exact revision before scaling.

| Item | Estimate / formula |
|---|---|
| Gaussian parameters | Position 12 B + scale 12 B + rotation 16 B + opacity 4 B + SH/color (4–192 B) before alignment; use 64–256 B/Gaussian for planning. |
| Optimizer state | Approximately 2–4× parameter bytes for Adam moments and gradients; measure for the chosen representation. |
| 1M Gaussians | Roughly 64–256 MB parameters, commonly 3–5× higher with gradients, moments, sorting keys, and tile buffers. |
| Render buffers | `width × height × (channels + auxiliary buffers) × bytes`; backward may require transmittance, IDs, depths, and gradients. |
| T4 constraint | 16 GB class; start at low resolution and low Gaussian count, use fp16/bf16 only after fp32 correctness. |
| MI300X constraint | Reported ~196 GB/device historically; do not treat capacity as permission to scale. |
| Dataset/artifacts | Keep raw datasets outside Git; record byte size, download time, checksum, and durable output location in the manifest. |

## Reproducibility and persistence

Every notebook or GPU job must persist before runtime expiry:

- pinned model and dataset revision;
- environment lockfile/container digest and compiler flags;
- JSON/YAML configuration and random seeds;
- input manifest with source URLs, licenses, and SHA-256 hashes;
- checkpoints at a fixed cadence plus the latest successful checkpoint;
- metrics, logs, provider receipt, and failure reason;
- artifact manifest linking parent inputs, software revision, and output hashes.

Colab/Kaggle local disks are caches, not durable storage. Upload only permitted checkpoints and small artifacts to a durable, access-controlled destination. Never upload secrets, virtual environments, or machine-specific binaries.

## Stop/go review template

Before approving the next tranche, answer:

1. What hypothesis did the last run test?
2. What was the actual billed cost and GPU count?
3. Did the output satisfy quantitative acceptance criteria?
4. What failure or uncertainty remains?
5. Why is the next experiment more informative than another CPU/T4 run?
6. What is its maximum cost, timeout, output, and stop condition?
7. Does the protected reserve remain at least 8 GPU-hours?

## Highest-information next action

The single next action is **read-only provider inventory plus a CPU/T4 renderer fixture**, followed—only after the inventory verifies one-GPU billing—by E0 capped at 1.5 billed GPU-hours. Additional MI300X use is justified only when E0 passes compile, numerical, gradient, determinism, and cost gates and a subsequent experiment names one measured bottleneck that cannot be resolved on CPU or T4.

## Sources

- AMD ROCm GSplat documentation: <https://rocm.docs.amd.com/projects/gsplat/en/latest/> (accessed 2026-10-11; compatibility and license must be rechecked at implementation time).
- Mip-NeRF 360 dataset page: <https://jonbarron.info/mipnerf360/> (accessed 2026-10-11; use restrictions and download details must be recorded in the dataset manifest).
- Engineering source repository metadata: <https://api.github.com/repos/kmmohaimenulhaque-code/BERKELIUM-NEURAL-RENDERING-AND-SCIENCES> (accessed 2026-10-11; public, default branch `main`, no declared GitHub license).
