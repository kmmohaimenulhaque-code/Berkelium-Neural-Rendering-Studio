# HANDOFF: Berkelium Neural Rendering Studio

Branch `Berkelium-Neural-Rendering-Studio-neural-v1`. Written 2026-10-17. It replaces the 2026-10-16 handoff;
that work is done and summarised in §5. Agent rules are in `CLAUDE.md`.

## 0. universal-cadquery (branch `universal-cadquery`, 2026-10-17): text → CadQuery on the MI300X
The owner runs this; agents cannot reach the droplet. Pipeline: text → Qwen3 + CadQuery LoRA → CadQuery script →
sandbox → STEP + GLB → core validation (repair ≤ 2 rounds) → mesh → existing Gaussian fit → AMD gsplat render
(`scripts/setup_rocm.sh`, gsplat `ff8bc54`) → Studio. Rule: **D-034** (pre-registered in its own commit `ff8a982`
before any training code existed).

**Run (on the droplet):**
```bash
export HF_TOKEN=...      # owner fills in; never printed or written to disk
curl -fsSL https://raw.githubusercontent.com/enigmahacker/Berkelium-Neural-Rendering-Studio/universal-cadquery/scripts/mi300x_oneshot.sh -o oneshot.sh && bash oneshot.sh
tail -f ~/berkelium_run/logs/latest.log           # watch
kill "$(cat ~/berkelium_run/run.pid)"              # stop; re-run the same bash command to resume
```
Stages (markers in `~/berkelium_run/markers/`, logs in `~/berkelium_run/logs/<stage>.log`): checks (rocm-smi sees
an MI300X, ROCm torch, Unsloth import, ≥ 500 GB free, HF_TOKEN) → env (venv on the image's torch,
`--system-site-packages`, image torch/vllm/unsloth/transformers frozen by constraints) → smoke (CPU tests,
tiny dataset, Qwen3-0.6B 20 steps) → data → D-034 gate (refuses to train if D-034 is not committed in the
branch history) → train → eval → publish → summary (paths, HF links, D-034 PASS/FAIL).

**Timeline (estimates, nothing measured on the MI300X yet):** setup ~0.5 h, smoke ~0.5 h, data 3–6 h
(`GEN_HOURS=2` of vLLM text generation for CAD-Recode), train: 32B probe ~1 h then one epoch on 32B or 14B
(the script measures tokens/s after 200 steps and switches to 14B if the projected epoch is > 10 h; written to
`~/berkelium_run/size_decision.json`), eval 2–4 h, publish ~1 h. Plan for 1–2 days.

**After training: plug the model into Studio (three modes), opt-in until D-034 passes**
(generator `cadquery`, or `BERKELIUM_CAD_BACKEND=cadquery` for the `cad` generator):
1. Endpoint: `BERKELIUM_CAD_URL=https://<vllm-or-HF-endpoint>/v1 BERKELIUM_CAD_MODEL=<served name>` (vLLM:
   `vllm serve Qwen/Qwen3-14B --enable-lora --lora-modules cadquery=<adapter>`; HF Inference Endpoint on the
   `-merged` repo). Key from `BERKELIUM_CAD_KEY` or `HF_TOKEN`.
2. Base + adapter on a local GPU: `BERKELIUM_CAD_BASE=Qwen/Qwen3-14B BERKELIUM_CAD_ADAPTER=Mhaquehaque/berkelium-qwen3-14b-cadquery-v1`.
3. Merged model on a local GPU: `BERKELIUM_CAD_BASE=Mhaquehaque/berkelium-qwen3-14b-cadquery-v1-merged`.

**Files:** `cadquery/` (sandbox, transpile, core_check, cqformat, cqpipeline, llm, pins, unsafe_corpus, tests),
`scripts/cq_data.py`, `cq_train.py`, `cq_eval.py`, `cq_publish.py`, `mi300x_oneshot.sh`,
`services/neural/berkelium_neural/cad/cadquery_backend.py`. Results land in `results/cadquery/E-C2.{json,report.md}`
and `publish_receipt.json`; `cq_publish.py` commits them on the droplet and writes `~/berkelium_run/results.bundle`
(push only with `BERKELIUM_GIT_PUSH=1`).

**Tested on CPU here (agent host, Python 3.12, cadquery 2.8.0, core `737dc35`):** `pytest cadquery/tests` and
`services/neural/tests/test_cadquery_backend.py` + `test_cad.py` pass: 73 tests, 92 s. Covered:
sandbox static check + isolated runs, all 42 injected unsafe scripts rejected with no side effect, timeout and
memory limits; transpiler round trip on every family (volume and bbox equal to the core's OCCT solid); exact
IoU; eval scoring with stub models; trainer encode/pack; data cleaning and repair corruptions; `cq_data.py --smoke`
on 60 real Text-to-CadQuery rows (20 per split; 60/60 verified after the clean-up) + families + repair; the Studio backend with
a stub model including a repair round; `mi300x_oneshot.sh` detaches and stops at the ROCm check on a non-ROCm host.
The rest of `services/neural/tests` could not be collected here (no torch/fastapi in this env; same before and
after the change).

**Not tested anywhere yet (ROCm/GPU):** Unsloth on ROCm and the PEFT fallback, vLLM-ROCm generation (and its HF
fallback), the 32B→14B size probe, checkpoint/resume under a real run, bf16 NaN behaviour, `unshare -rn` network
isolation inside the droplet's container (falls back to an in-process socket block; recorded), the full-size
downloads (CAD-Recode is one shallow git clone with a 20 min budget), `cq_publish.py` uploads, and the merged
model. No E-C2 number exists until the owner's run writes `results/cadquery/E-C2.json`.

**Licences:** CAD-Recode is **CC-BY-NC-4.0**, so the E-C2 model is treated as non-commercial; Text-to-CadQuery
states no licence (derived from Text2CAD/DeepCAD). See `THIRD_PARTY_NOTICES.md`.

### 0.1 Live run status (2026-10-17, ~22:00Z): see `1AMmi300x.md` for the full log
- Data done (220 min): 121,575 train rows, 74.8M tokens. The D-034 gate passed.
- **Training: Qwen3-32B + Unsloth, LoRA r=64, 2,780 steps, all of them** (owner kept 32B over the 10 h size rule:
  `--no-size-switch`). Last reported: step 508/2780 after 2:13 h, 15.67 s/step, about 9.9 h left (tqdm estimate).
- **Eval skipped (owner decision, deadline).** D-034 is not applied. The model ships **opt-in, "not evaluated"**, and
  the cards say so. There are no E-C2 numbers.
- **Automatic after training** (no laptop needed; the jobs run detached on the droplet):
  1. `publish`: the merged bf16 model is built on CPU and uploaded (public, `PUBLISH_ARGS=--public`):
     - `Mhaquehaque/berkelium-qwen3-32b-cadquery-v1`: LoRA adapter (~2 GB), run manifest, card
     - `Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged`: full bf16 safetensors (~65 GB)
  2. `scripts/cq_gguf.sh` (already running, waiting for `publish.done`): llama.cpp `620d222` →
     `Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-GGUF` with `*-Q4_K_M.gguf` (~20 GB, estimate) plus Ollama
     `system`/`params` files, for `ollama run hf.co/Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-GGUF:Q4_K_M`.
  3. `results/cadquery/HANDOFF_TEAMMATE.md` on the droplet: links, base model + revision, steps trained, measured
     file sizes, full weights vs LoRA, GGUF sha256, licence (non-commercial, CAD-Recode CC-BY-NC-4.0).
- No weights in git (teammate request = CLAUDE.md rule 7). Only small JSON/MD files are committed on the droplet.
- Untested until it runs: the merge on CPU, the HF uploads, the llama.cpp convert of this model, and Ollama reading
  `system`/`params` from the repo. If the GGUF fails, the `-merged` repo is the fallback for the teammate.
- The pipeline's final summary step will crash after publish (no `E-C2.json`). This is expected and harmless.

## 1. Direction (owner decision, 2026-10-17)
Go all in on **Qwen3-8B on Kaggle** as the text → design compiler. Never on the agent host's CPU, never MI300X.
The pipeline is: any natural text → `DesignProposal` JSON (gear/beam CEM requirements, or procedural Geometry
IR, the core's CadQuery-like op graph) → Berkelium core (exact STEP via OCCT / mesh via Manifold, validation
L0–L6) → GLB → Gaussians → render.

Why: the core repo already does the deterministic half (schemas, Geometry IR, OCCT, gear + beam CEMs,
validation, repair loop). What's missing:
1. A model that writes valid designs for **any** object. The owner's M1 adapter (Qwen3-32B) knows only gears
   plus 30 lanterns.
2. Any evaluation of that model at all. None of the 4 HF adapters has been evaluated (`ENDGAME_AUDIT.md` §4
   in the core repo).
3. The link between the core and Studio.

## 2. Inputs on record
| Item | Where | Notes |
|---|---|---|
| Core (deterministic engine) | github `kmmohaimenulhaque-code/BERKELIUM-NEURAL-RENDERING-AND-SCIENCES` @ `physics/p1-numerical` `737dc35` | Geometry IR ops: box, cylinder, cone, sphere, torus, profile (line/arc/spline), circle, extrude, revolve, union/difference/intersection, translate/rotate/mirror, polar/linear pattern, implicit; expressions over parameters |
| Gear dataset | HF `Mhaquehaque/berkelium-engineering-dataset` | 574 train rows: gear plan/repair + 30 held-out lanterns |
| 32B adapters | HF `Mhaquehaque/berkelium-qwen3-32b-{lora,agent-v1,agent-v2,agent-v2-dpo}` | M1 = text→design (gear only). V1/V2/DPO = agent policies (ClaimEnv/ModelEnv), not text→geometry. **All unevaluated.** Not used by this plan (32B doesn't fit a T4). |
| Kaggle | account `kmmohaimenulhaque`, API checked working 2026-10-17 | T4 16 GB (×1 or ×2), 12 h per session, ~30 h/week |

## 3. Checklist
Status: ☐ todo · ◐ in progress · ☑ done (with evidence file)

### K0 — Setup
- ☑ Core reviewed and its install checked (OCP + manifold import OK). Its full test suite was not run here
  (the owner stopped CPU runs).
- ☑ Kaggle API reachable (`kaggle kernels list --mine` OK, 2026-10-17).
- ☑ Kaggle spec `configs/kaggle/cad_dataset.json` installs core `@737dc35[occt,mesh]` inside the kernel (CPU-only kernel, no GPU quota).

### K1 — Dataset `berkelium-cad-v1` (built and verified **on Kaggle**) ☑ 2026-10-17
**Result:** 19,710 candidates → 18,712 unique verified rows (train 16,929 · val 540 · test 889 · held-out 354), 866 rejected with reasons, 13 % vague. Evidence: `results/cad/cad_v1_manifest.json`; file hashes pinned in `configs/datasets/cad_v1.sha256.json`; Kaggle receipt `results/kaggle/cad-dataset/20261017T084505Z/receipt.json`.
- ☐ `scripts/cad_dataset.py`: ~35 parametric families. Each emits a DesignProposal (named parameters plus an
  expression-based Geometry IR) and several natural-language renderings: precise/vague, mm/cm/inch, terse or
  descriptive, and vague versions that fill `assumptions`/`open_questions`. Families:
  - **Mechanical parts:** plate with hole pattern, L-bracket, U-channel, I-beam, flange with bolt circle,
    washer, spacer, stepped shaft, hex nut, hex bolt, pulley, spoked wheel, pillow block, sprocket-like toothed
    disk, hinge leaf, knob, heat sink, enclosure + lid, honeycomb panel.
  - **Piping and vessels:** pipe tee, pipe elbow, pressure vessel, funnel.
  - **Aerospace:** rocket nozzle (converging-diverging spline, revolved), rocket body (ogive nose + tube + fins),
    propeller/fan.
  - **Everyday objects:** mug, bottle, vase, table, chair, ladder, truss, lantern.
  - **Lattice:** gyroid (implicit).
  - **Engineering models:** gear pair (from the core factory), cantilever beam (CEM).
  - **Compositional grammar:** a base body plus N features (holes, pockets, bosses, patterns), with text
    generated from the feature list, for open-ended variety.
- ☐ Every row is realised by the core (L3 pass, requirement checks pass). Rejected rows are logged with the reason.
- ☐ Splits are by family and seed. **Held-out families** (never trained on) measure generalisation to unseen
  objects. Target: ~15–25k verified rows.
- ☐ Publish as a Kaggle Dataset (+ HF `Mhaquehaque/berkelium-cad-v1` if an `HF_TOKEN` secret is set).
  Commit the manifest (hashes, counts) to `results/cad/`.

### K2 — Pre-register, then train Qwen3-8B
- ☑ **D-033 pre-registered in commit `eb7764e`, before training.** M2 (Qwen3-8B + LoRA) is promoted to the Studio `cad` generator only if,
  on held-out data:
  - in-distribution test: schema-valid ≥ 95 %, realise (L3) ≥ 90 %, requirement pass ≥ 85 %;
  - held-out families: realise ≥ 60 %, and mean voxel IoU against the reference beats zero-shot Qwen3-8B
    (paired 95 % CI above 0);
  - gear test (core `berkelium eval` metrics): not worse than zero-shot Qwen3-8B;
  - 0 proposals that contain measurement or validation fields (schema `extra=forbid` rejects them).
- ◐ `configs/kaggle/cad_train.json` launched 2026-10-17 (kernel `kmmohaimenulhaque/berkelium-qwen3-8b-cad-train`; the first kernel slug sat in the T4 queue 40 min and was deleted, 2026-10-17): Qwen3-0.6B smoke, then Qwen3-8B, 10 h budget. QLoRA: NF4 base, fp16 compute (the T4 has no bf16), LoRA r=32 on
  all linear layers, seq ≤ 3072, gradient checkpointing, checkpoint every 30 min, resume across 12 h sessions,
  NaN guard. Compact JSON targets (sorted keys, no spaces).
- ☐ Adapter goes to HF `Mhaquehaque/berkelium-qwen3-8b-cad-v1` with a model card + run manifest (data/config
  SHA-256).

### K3 — Evaluate on Kaggle
- ☐ One eval job `configs/kaggle/cad_eval.json` (kernel `berkelium-cad-eval`): zero-shot on GPU 0 and m2 on GPU 1 in parallel, then core scoring + D-033. (The separate baseline kernel sat in the T4 queue 75 min and was deleted unrun, 2026-10-17.) HF transformers, NF4 (same quantisation as training), greedy, one shard per T4.
  Generate for zero-shot Qwen3-8B and M2 on test + held-out families + gear test.
- ☐ Score inside the kernel with the core: parse → schema → realise → requirements → voxel IoU. Repair loop
  (≤ 3 JSON-Patch rounds) is reported separately, as "with repair".
- ☑ **E-C1 done 2026-10-18** (`results/cad/E-C1/E-C1.report.md`): m2 test realise 0.998 / req 0.997 / IoU 0.936 (base 0.00);
  held-out families realise 0.49 (needs ≥ 0.60) → **D-033 FAIL, not promoted; opt-in only.** Next: add more varied families
  + paraphrased text, retrain (the 32B CadQuery run on the MI300X is the broader attempt).

### K4 — Studio integration
- ☑ `cad` generator (`berkelium_neural/cad/generator.py`, tests `services/neural/tests/test_cad.py` 12 passed): text → M2 (Kaggle/HF endpoint or local GPU) → core `run()` → GLB → mesh → Gaussians → render.
  `open_questions` are shown to the user instead of guessing. The provenance card shows the proposal, the
  validation report, artifact hashes and the adapter hash.
- ☐ Router: an engineering intent goes to `cad`; organic objects go to `auto` open-world, labelled.
- ☐ Tests use a stub model (no GPU in CI). Docs: README, ROADMAP, ENGINEERING_LOG, THIRD_PARTY_NOTICES (Qwen3
  Apache-2.0; OCCT LGPL; core license still undecided, U8).

### Budget (Kaggle T4 hours, estimates)
K1 build ≈ 1–2 h (CPU-heavy, run inside a Kaggle session) · K2 train ≈ 6–10 h (two sessions) · K3 eval ≈ 2–3 h.
Total ≈ 10–15 h of the weekly ~30 h.

### Studio readiness (2026-10-18)
- ☑ `cad` (8B) runs end to end through the real service: job → stub compiler → core geometry → mesh → Gaussians → splat
  render, with `design_proposal.json` + `core_validation.json` on the mesh artifact
  (`services/neural/tests/test_openworld_service.py::test_cad_job_runs_core_geometry_through_gaussians_and_render`).
  Neural suite: 140 passed, 12 skipped (GPU / real-CadQuery), `test_durable_jobs.py` not re-run.
- To use it with the real model: install the core (`pip install -r configs/environments/berkelium-core.txt`), then either
  `ollama run hf.co/Mhaquehaque/berkelium-qwen3-8b-cad-v1-GGUF` + `BERKELIUM_CAD_URL=http://localhost:11434/v1`
  `BERKELIUM_CAD_MODEL=hf.co/Mhaquehaque/berkelium-qwen3-8b-cad-v1-GGUF`, or `BERKELIUM_CAD_ADAPTER=Mhaquehaque/berkelium-qwen3-8b-cad-v1`
  on a CUDA host. Pick **Engineering CAD** in Generate. Not yet run with the real model in Studio (no GPU in agent sessions).
- Known clash: the repo-root `cadquery/` folder (E-C2) has the same name as the CadQuery library. Never put the repo root
  first on `sys.path` (fixed in `test_benchmark_reporting.py`).

## 4. Risks
- **fp16 overflow** in Qwen3 on the T4. Mitigation: NF4 + fp16 compute with fp32 LoRA; NaN guard aborts and
  records the run.
- **"Any text" is bounded by the IR + data.** There are no fillets, chamfers, threads or sweeps along paths, so
  such requests get approximations plus an `assumptions` entry. Held-out-family IoU is the honest measure of reach.
- Templated text can be memorised. Defence: varied renderings, held-out families, and real user-like prompts in
  a small hand-written test set (stated as such).

## 5. Earlier results (2026-10-16, unchanged)
E-T3: Shap-E B1 F@0.05 0.669 / B2 R@1 0.95 (the `auto` default, D-029). zimage-triposr v2 0.603 / 0.85, best
on B4 composition. Anchor LoRA and capsule v2 were not promoted (D-030, E-T4). Z-Image-Turbo was fully
downloaded (30.59 GB, SHA-256 verified) and ran on a T4 (`results/m5/E-A1_receipt.json`). Details are in
`README.md`, `results/text3d/*.report.md` and `docs/ENGINEERING_LOG.md`.

## 6. Delivery
Pushing from the agent returns 403 until the Claude GitHub App has access to this repo
(https://claude.ai/connect-github). Otherwise, one small bundle goes to the owner. Never upload weights or
chunks to git.
