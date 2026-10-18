# Environment specifications

Three environments, never mixed (a ROCm PyTorch must not have CUDA runtime wheels next to it, and vice versa —
`scripts/doctor.py` flags it).

| file | purpose | evidence it produces |
|---|---|---|
| `cpu.txt` | CPU reference, HIP-CPU kernel validation, text-to-3D training/eval, API, Studio E2E | CPU semantics + kernel logic (NOT GPU execution) |
| `rocm-7.0-pt2.8-py3.12.txt` | AMD gsplat baseline + Berkelium HIP kernel on MI300X (gfx942) | real HIP execution |
| `rocm-6.4.3-pt2.6-py3.10.txt` | same, older AMD-supported pair | real HIP execution |
| `t4-cuda.txt` | Kaggle/Colab T4 experiments with upstream (nerfstudio) gsplat on CUDA | CUDA reference only — does NOT validate HIP |

`cpu.txt` pins the exact versions on which the CPU test suite in this commit was run (Python 3.13, 2026-10-13).
The ROCm/CUDA files pin only what AMD/PyTorch publish as compatible; they were **not executed** in this commit.
AMD support matrix source: AMD-Ecosystem/gsplat tag v1.5.3b2 README (ROCm 6.4.3 / 7.0.0, PyTorch 2.6 / 2.8,
Python 3.10 / 3.12, Ubuntu 22.04 / 24.04, MI300X).
