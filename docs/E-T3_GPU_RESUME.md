# E-T3 on a GPU: resume the frozen benchmark (MI300X first, Kaggle 2×T4 only if needed)

Status when written (2026-10-16 03:30 UTC): **no GPU unit has run.** The development Codespace has no credentials
for the MI300X or Kaggle. Both platforms are run by the owner with the commands below. Everything here is
resumable, and nothing re-runs a finished unit.

## What is already done (CPU, this Codespace)

- `results/text3d/E-T3/units.jsonl` is the CPU ledger. It holds all 56 native units and the Shap-E units of the CPU
  run (B1 seed 0, then B4 single-shot, then B2). That run continues until its frozen budget of 28 Shap-E units or
  its 7.5 h wall deadline, about 17.5 min per unit.
- The full frozen design has 124 units (56 native + 68 Shap-E). The GPU's share is whatever the CPU run will not
  do:

| Group | Units | Who |
|---|---|---|
| B4_planner (Shap-E generates each planner part; 16 distinct parts) | 16 | **GPU** (outside the CPU budget) |
| B1 seed 1 and seed 2 (variance) | 24 | **GPU** |
| B1 / B4 single-shot / B2 seed-0 units the CPU run did not reach | 0-4 expected | GPU with `--include-cpu-reserved`, **only after the CPU run has stopped and its ledger is pushed** |

## How a GPU unit stays comparable with a CPU unit

Same weights (sha256-checked), same 16 Karras steps, guidance 15, **fp32 sampling** (`--fp32`), and the **same
initial noise**: Shap-E's Heun sampler with `s_churn=0` is deterministic given `x_T`, and on a GPU `x_T` is drawn
from the seeded CPU generator and moved to the device (`shap_e_backend._cpu_noise`, unit-tested). GPU kernels
still differ numerically from CPU ones, so results are not bitwise equal. Every ledger row and report row records
its device. The smoke test measures the CPU/GPU difference on the same draw before anything else runs.

## MI300X (first choice)

```bash
cd ~/Berkelium-Neural-Rendering-Studio && git pull origin Berkelium-Neural-Rendering-Studio-neural-v1
source <the ROCm venv used for E-R1>/bin/activate          # torch 2.8.0+rocm7.0.2, Python 3.12
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"   # hip set, True
PYTHON=$(which python) bash scripts/setup_openworld.sh      # clip + shap-e (--no-deps) into this venv; ~3.1 GB of
                                                            # weights into data/models/shap_e, sha256-verified
# 1. gate (~1 GPU allocation, one generation). Writes results/text3d/E-T3/gpu_smoke/<host>_cuda/smoke.json
python scripts/run_openworld_benchmark.py gpu-smoke --device cuda
# 2. only if the gate passed: the 40 units outside the CPU budget (stops at --wall-hours)
python scripts/run_openworld_benchmark.py generate --device cuda --wall-hours 3
# 3. after the CPU run has stopped and its ledger is pushed: the seed-0 units it did not reach
git pull && python scripts/run_openworld_benchmark.py generate --device cuda --include-cpu-reserved --wall-hours 1
# 4. bring the results back
git add results/text3d/E-T3 && git commit -m "E-T3: GPU units (MI300X)" && git push
```

The smoke gate passes if the mesh is valid. `smoke.json` also reports `vs_cpu_same_draw`: Chamfer-L1 in the
shared frame against the CPU chair of the same seed (`equivalent_to_cpu` if < 0.01), the measured seconds per
unit, peak GPU memory, and the runtime estimate for the pending units. If it fails, keep `smoke.json` and the
unit's `.log` (full traceback). Make at most two targeted fixes; if it is still blocked, use Kaggle. Likely
fault points: ROCm fp32 matmul paths and Shap-E's torch marching cubes on device. Weights are loaded to CPU and
copied, so mmap is not used there.

## Kaggle, 2 × T4 (only if the MI300X is blocked or too slow per measured throughput)

Notebook settings: accelerator "GPU T4 x2", internet on. Weekly quota: check the notebook's quota panel first.

```bash
!git clone -b Berkelium-Neural-Rendering-Studio-neural-v1 https://github.com/enigmahacker/Berkelium-Neural-Rendering-Studio.git
%cd Berkelium-Neural-Rendering-Studio      # private repo: clone with a token or upload the repo as a Kaggle dataset
!pip install -q trimesh scikit-image pydantic
!PYTHON=python bash scripts/setup_openworld.sh
!python scripts/run_openworld_benchmark.py gpu-smoke --device cuda
# only if the gate passed. Two T4s do not speed up one generation; two independent shards may raise throughput.
# Start with one shard; add the second only if host RAM allows (each worker ~2.4 GB) and the measured rate improves:
!CUDA_VISIBLE_DEVICES=0 nohup python scripts/run_openworld_benchmark.py generate --device cuda --shard 0/2 --wall-hours 3 > shard0.log 2>&1 &
!CUDA_VISIBLE_DEVICES=1 nohup python scripts/run_openworld_benchmark.py generate --device cuda --shard 1/2 --wall-hours 3 > shard1.log 2>&1 &
# before the session ends: export (ledgers + unit folders), then commit them in the repo
!zip -qr /kaggle/working/et3_gpu.zip results/text3d/E-T3/units_cuda_* results/text3d/E-T3/units results/text3d/E-T3/gpu_smoke
```

Each shard writes its own ledger (`units_cuda_<host>_shard<i>of<n>.jsonl`). Shards are disjoint by construction,
and every ledger row says which GPU it ran on.

## After the GPU units are back (CPU Codespace)

```bash
git pull
.venv-cpu/bin/python scripts/run_openworld_benchmark.py score     # reads every units*.jsonl; duplicates are reported
.venv-cpu/bin/python scripts/run_openworld_benchmark.py report    # E-T3_openworld_cpu.report.md, per-device cost table
```

## Budget

Confirmed by the owner: ~24 MI300X GPU-hours and ~30 Kaggle T4 GPU-hours this week. Rates are not verified here,
and no throughput figure is assumed: the smoke run measures seconds per unit, and `smoke.json` estimates the hours
for the pending units. Stop and reassess if the measured seconds per unit times the pending units exceeds 3 GPU-hours.
