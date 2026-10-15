# E-T3 P2: one real Shap-E generation on this host, and the benchmark budget derived from it (2026-10-15)

Run: `/usr/bin/time -v .venv-cpu/bin/python scripts/openworld_generate.py "a wooden chair with four legs" --steps 16
--threads 2` in its own process. CPU only (2 vCPU, no GPU). Outputs: `p2/a_wooden_chair_with_four_legs__s0/`
(`mesh.glb`, `preview.png`, `report.json`, `time_v.txt`).

| Measure | Value |
|---|---|
| Result | real mesh, recognisably a wooden chair with four legs, seat, back and stretchers, upright in +Y (`preview.png`) |
| Mesh | 19,540 vertices, 39,094 faces, 1 component, edge-manifold, winding-consistent, not watertight |
| Validity as recorded | **invalid**: 1 zero-area face. Cause: a bug in `finalize_mesh` (cleaned in float64, then cast to float32, so a 1e-12 face rounded to zero). Fixed (round first, then clean) with a regression test. Re-cleaning this same GLB with the fixed order removes that one face: valid, 39,093 faces |
| Load (text300M + CLIP text tower, memory-mapped) | 19 s |
| Sampling (16 Karras steps, Heun, CFG 15) | 763.0 s = 31 denoiser calls x 24.6 s = **47.7 s per step** |
| Decoding (STF mesh, checkpoint grid) | 40.8 s |
| In-process total | 804.9 s |
| Wall clock (`time -v`) | **827 s (13 min 47 s)** per generation, including interpreter start |
| CPU | 1,177 s user + 184 s system, 164% of 2 vCPU |
| Peak RSS | 2,597 MiB (`/proc` sampler: 2,287 MiB anonymous + 1,241 MiB file-backed peaks) / 2,609 MiB (`time -v`); watchdog kills at ~3.2 GB, self-abort guard 3,100 MiB: **not triggered** |
| Fallbacks (fp16 CLIP text, lower decoder grid) | not needed |
| Disk | Shap-E + CLIP weights 3.42 GB (`data/models/shap_e`: text_cond 1.26 GB, vector_decoder 0.91 GB, ViT-L-14 0.93 GB, extracted text tower 0.32 GB); evaluator CLIP ViT-B/32 0.35 GB; ~1-3 MB per generated mesh; 5.9 GB free after all downloads |

## Budget (computed before any benchmark generation)

Available CPU time for benchmark generation on this host: **6.5 hours** (the rest of the sprint's wall clock is
for integration, tests and the E2E run, which need the same 2 vCPUs and cannot overlap a Shap-E run on 7.8 GB RAM).

    N_shap_e = floor(6.5 h x 3600 / 827 s) = floor(28.3) = 28   (cap 40)

Generation calls the full E-T3 design would need (unique (generator, prompt, seed) units; auto reuses the unit of
the generator it routes to; planner parts are generated once each; failed generations are results, never retried):

| Block | Shap-E units | Native units |
|---|---|---|
| B1: 12 GSO prompts | 12 | 12 |
| B2: 8 GT-free prompts (the B1 prompts reuse B1 units) | 8 | 8 |
| B3: 12 procedural prompts (auto routes all 12 to native) | 0 | 12 |
| B4 single-shot: 8 two-object prompts | 8 | 8 |
| B4 planner: 16 distinct part prompts | 16 | 16 |
| B1 variance, seeds 1-2 | 24 | 0 |
| E2E (P5): 1 full + 1 cancelled mid-sampling | ~1.1 | 0 |
| **Full design** | **69** (~15.9 h) | 56 (minutes) |

**Minimum benchmark that fits N = 28 (run in this priority order):** all native units, B1 Shap-E (12), B4
single-shot Shap-E (8), B2 Shap-E (8). **Omitted and labelled in the results:** the B4 Shap-E-planner arm (16 units)
and the B1 seed-1/2 variance runs (24 units). The E2E generation runs outside this budget.
The results will therefore be reported as a partial E-T3, never as the complete design.

## Update after the first service E2E runs (same day)

- The host's watchdog acts on **system-wide** available memory, not on one process's RSS. Every SIGTERM so far
  happened when MemAvailable fell to ~0.7-0.8 GB: 7.8 GB total, ~4 GB used by the editor/agent processes, plus
  the process that was killed. In the E2E, a second Shap-E worker started after the first job was SIGTERMed 12.5 s
  into loading (exit -15). Hypothesis tested and rejected: the backend's splat render bloats the API process (it
  peaks at 346 MB).
- Fix: Shap-E models are now built with `torch.nn.init` disabled. Their parameter buffers are never written, so
  never resident, and the memory-mapped checkpoint is assigned in their place (strict; the build is refused if
  the checkpoint does not match exactly). Measured loading peak: text300M **1.49 GB RSS / 1.09 GB anonymous**
  (P2: 2.29 GB anonymous peak), decoder 0.32 GB. Weights stay file-backed (reclaimable page cache).
- The service now refuses to start a worker when MemAvailable < 2,000 MiB (`RESOURCE_LIMIT`, configurable via
  `BERKELIUM_SHAP_E_MIN_FREE_MIB`). It reports a signal-killed worker as such instead of a generic failure.
