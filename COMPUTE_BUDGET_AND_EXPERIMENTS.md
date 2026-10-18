# Compute Budget and Experiments

**Status:** architecture and experiment-control plan, revision `NR-0.2` (2026-10-12); no GPU work authorized by this document.

## Executive rule

The reported MI300X balance is approximately **27 GPU-hours**, but it is not yet a verified balance. No MI300X execution is permitted until the provider confirms the billing unit, allocated GPU count, visible device count, quota, and remaining balance. A one-hour run on a four-GPU allocation may consume four GPU-hours or more depending on the provider's accounting. [D] The proposed first tranche is **at most 1.5 billed GPU-hours**, one GPU only, followed by a go/no-go review; protect **at least 8 billed GPU-hours**. These ceilings are not execution authorization.

Evidence convention: `[O]` inspected implementation; `[F]` external source fact; `[D]` proposed decision; `[H]` untested hypothesis. Resource reports below are historical/unverified, not a live inventory.

## Confirmed, reported, and unknown resources

| Resource | Current evidence | Planning meaning |
|---|---|---|
| AMD MI300X | User reports about 27 remaining free hours; earlier diagnostics reported Ubuntu 24.04, ROCm 7.0, PyTorch `2.10.0+rocm7.0`, HIP `7.0.51831`, `gfx942`, and about 196 GB visible memory. | Treat all values as historical until a read-only inventory confirms them. Use one GPU by default; do not treat nominal memory as a scaling target. |
| Kaggle T4 | User reports access to T4-class notebooks. | CUDA/PyTorch reference work, preprocessing, small inference, and evaluation only; verify actual allocation and session limits each run. |
| Colab T4 | User reports access to T4 notebooks. | Independent compatibility and inference checks; never assume availability or durable runtime. |
| Hugging Face | Hub/model/dataset access is available or intended. | Distribution, pinned revisions, metadata, and artifacts; hosted GPU is optional and must be cost/quota-verified. |
| Target repository | Local extracted Studio source is present, but this workspace has no `.git` metadata; GitHub target returned 404 to unauthenticated inspection. | Do not claim branch/working-tree state for the remote target. Preserve local files and request authenticated access before remote changes. |
| Engineering source | Inspected checkout is revision `737dc35c42aaf97f52a6ac984a806adef9518525` on `origin/physics/p1-numerical`; remote `main` is `70cc43ee41b8dec2de515be35d27e98b952337b`; README licensing remains unresolved. | Reference/integration source only. No code transfer until file-level licensing and ownership are cleared. |

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

**Experiment E-A0 — single-scene, low-resolution renderer and ROCm feasibility gate.** The CPU contract/gradient fixture must pass before this allocation.

Before MI300X use, pass CPU analytic and finite-difference fixtures on owned synthetic Gaussians. No dataset or weights are needed. E-A0 is the allocation envelope for renderer experiment **E0** in [TEXT_TO_3D_RESEARCH_PLAN.md](TEXT_TO_3D_RESEARCH_PLAN.md), not a competing experiment definition. Mip-NeRF 360 is a later independently licensed real-scene evaluation, not a prerequisite.

| Item | Budget and method |
|---|---|
| MI300X allocation | 1 GPU, maximum 1.5 billed GPU-hours; hard timeout at 90 minutes wall time unless provider billing is not GPU-hour based, in which case stop and recalculate. |
| Workload | Research E0: 64-Gaussian synthetic recovery, 128²/12 cameras/200 iterations maximum; 10k/256² and 100k/512² timing fixtures. Maximum 30 minutes from first GPU operation, within the allocation ceiling including setup/teardown. |
| Expected outputs | Environment manifest, kernel build log, rendered PNG/NPZ, loss curve, forward/backward timings, peak VRAM, Gaussian count, checksum, and failure classification. |
| Acceptance | Pinned build; renderer G1/G2 tolerances for every parameter, not just gradient signs; 10-repeat determinism; finite outputs; E0 train loss reduction >=50% and holdout RGB MSE reduction >=20%; G4 timing/memory gates. Preserve correctness-pass/performance-stop as a partial result. |
| Stop condition | Stop immediately if the HIP extension cannot compile, the reference comparison diverges beyond tolerance, memory exceeds the measured envelope, or the provider reports unexpected multi-GPU billing. A failed hypothesis is useful; do not extend the run to obtain a better-looking image. |
| Go condition | Proceed only if all acceptance checks pass and the actual billed cost is within the approved tranche. Reassess with observed timings rather than authorizing a full run automatically. |

The first experiment answers the highest-risk question: whether the custom HIP path can execute a small, verifiable differentiable renderer on the actual AMD allocation. It does not claim reconstruction quality or text-to-3D capability.

## Cumulative budget ledger

| Experiment | Resource | Max billed GPU-hours | Actual GPU-hours | Gate |
|---|---|---:|---:|---|
| E-A0 HIP feasibility (research E0) | 1× MI300X | 1.5 | Not run | CPU G1 + explicit authorization + provider inventory |
| E-T0 optional CUDA control/metrics | Kaggle or Colab T4 | 0 MI300X | Not run | CPU fixtures available; verify notebook quota |
| E-R1 renderer benchmark (covers E-A0's HIP feasibility question), 2026-10-13 | 1× MI300X VF | — | not recorded from a provider receipt | completed: `results/renderer/E-R1_mi300x.20261013T200512Z.json` |
| E-R1 rerun incl. one failed attempt, 2026-10-15 | 1× MI300X VF | — | not recorded from a provider receipt | completed: `E-R1_mi300x.20261015T160753Z.json` (failed attempt kept) |
| E-A1 learned image smoke (research E1) | 1× MI300X | 1.0 | Not run (preflight blocked on the CPU dev host: `results/m5/E-A1_preflight.json`) | E0 pass; separate download/run permission and license/operator inventory |
| E-A2 HIP renderer scale test | 1× MI300X | 2.0 | Not run | E0 pass; target one bottleneck only |
| E-A3 small end-to-end scene optimization | 1× MI300X | 3.0 | Not run | E-A2 pass and quality baseline recorded |
| E-A4 integration/debug reserve | 1× MI300X | 4.0 | Not run | Only for a named blocker with reproducible failure |
| E-A5 final validation reserve | 1× MI300X | 4.0 | Not run | Only after API/artifact/renderer integration is testable |
| **Protected reserve** | 1× MI300X | **8.0 minimum** | `TBD` | Must not be allocated in advance |
| **Uncommitted remainder** | Provider-dependent | **3.5** | Not allocated | Conditional on exactly 27 GPU-hours verified |

The illustrative arithmetic is `1.5 + 1 + 2 + 3 + 4 + 4 + 8 + 3.5 = 27`. The ledger must be updated after every allocation from the provider receipt, not from wall-clock guesses. Setup, idle time, billing rounding and teardown count; release early enough to remain below the ceiling. If instance billing or minimum allocation changes the cost, stop and recalculate before starting. Protected reserve is not preapproved spending.

## Resource placement matrix

| Work | CPU | Kaggle/Colab T4 | MI300X/HIP | Hugging Face |
|---|---:|---:|---:|---:|
| Contracts, schemas, API tests | Primary | No | No | No |
| Camera validation and manifests | Primary | Optional | No | Dataset metadata only |
| CPU renderer/reference gradients | Primary | Optional | No | No |
| CUDA reference renderer | No | Primary | No | No |
| HIP kernel correctness | No | Cannot validate | Required | No |
| MI300X performance tuning | No | Cannot validate | Required, late | No |
| Z-Image smoke test | Operator/metadata audit | Optional compatibility check; 16 GB feasibility unverified | Research E1 only after E0 and separate approval | Model revision source |
| Full-resolution/multi-scene work | No by default | Not authorized by default | Not authorized by default | Artifact distribution only |

CUDA execution cannot validate HIP device correctness, and CPU agreement cannot establish AMD performance.

## Memory and storage estimates

These are planning estimates, not measurements. Measure them on the exact revision before scaling.

| Item | Estimate / formula |
|---|---|
| Gaussian parameters | Position 12 B + scale 12 B + rotation 16 B + opacity 4 B + SH/color (4–192 B) before alignment; use 64–256 B/Gaussian for planning. |
| Optimizer state | Approximately 2–4× parameter bytes for Adam moments and gradients; measure for the chosen representation. |
| 1M Gaussians | Renderer base example with 8M pairs and 1024² pixels: `236 MB + 80 MB + 896 MB + 41.94 MB = 1.254 GB decimal` (about 1.17 GiB), excluding tile metadata, alignment, checkpoints, library scratch and copies. Pair overlap dominates; use the renderer spec formula, not a parameter-only multiplier. |
| Z-Image-Turbo snapshot | Selected files approximately 32.83 GB on disk; not measured runtime VRAM. Allow separate disk space for caches/staging and do not infer T4 fit from a model headline. |
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

The next implementation action is the **CPU-only contract/gradient fixture**. A separately authorized read-only provider inventory precedes E-A0 (research E0), capped at 1.5 billed GPU-hours. Additional MI300X use requires a new bounded hypothesis, passing prerequisite gates, and cost review; no allocation is authorized by writing this plan.

## Sources

- AMD ROCm GSplat documentation: <https://rocm.docs.amd.com/projects/gsplat/en/latest/> (reviewed 2026-10-12; compatibility and license must be rechecked at implementation time).
- Mip-NeRF 360 dataset page: <https://jonbarron.info/mipnerf360/> (reviewed 2026-10-12; use restrictions and download details must be recorded in the dataset manifest).
- Engineering source branch/ownership evidence: [REPOSITORY_INTEGRATION_MAP.md](REPOSITORY_INTEGRATION_MAP.md); default-branch metadata is not evidence of the inspected feature branch's contents.
