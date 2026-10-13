# GPU run plan (requires owner approval — nothing here has run)

Rate and billing unit are NOT verified. Estimates assume one MI300X billed per GPU-hour; check the provider's
current rate before approving. All runs are resumable and write results/receipts even on failure.

| Run | What it proves | Command | Est. wall time | Est. GPU-hours |
|---|---|---|---|---|
| R0 setup + doctor | ROCm/PyTorch/gsplat tuple is real; no CUDA/ROCm mixing | `bash scripts/setup_rocm.sh` then `--install` | 15–25 min (mostly image pull / gsplat build) | ~0.4 |
| R1 smoke | gsplat + Berkelium kernel execute on gfx942 | `bash scripts/setup_rocm.sh --install --smoke` | 2–5 min (incl. hipcc JIT of the Berkelium kernel) | ~0.1 |
| R2 ROCm tests | gsplat forward parity vs FP64 reference; Berkelium vs gsplat vs reference gradients; 10-repeat bitwise determinism on GPU | `python -m pytest services/neural/tests -q -m hip` | ~5 min | ~0.1 |
| R3 E-R1 benchmark | matched gsplat vs Berkelium: correctness gates, median/p95 fwd/bwd/iteration latency, peak memory, workload, fit PSNR | `python scripts/run_benchmarks.py --experiment configs/experiments/E-R1_renderer_mi300x.json --i-approve-gpu-cost` | ~25 min | ~0.4 |
| **Total (R0–R3)** | | | **~50–60 min** | **~1.0** |

Optional, after R0–R3:

| R4 text-to-3D v2 | larger BerkeliumTextShape (hidden 512, 20k steps) on GPU; same protocol E-T1 | add `"device": "cuda"` support to train.py first (small change) | ~30–45 min | ~0.75 |

Recommended environment: the AMD image `rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0`
(PyTorch 2.8 is in AMD gsplat's support matrix; the historical MI300X environment with PyTorch 2.10 is not).
Do not pip-install a second PyTorch into an existing ROCm environment.

Stop conditions: any R2 failure stops R3 (investigate on CPU first; HIP-CPU reproduces kernel logic). Do not
loosen tolerances to pass. Delete the instance when R3's result file is written.
