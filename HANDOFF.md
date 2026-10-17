# HANDOFF: Berkelium Neural Rendering Studio

Branch `Berkelium-Neural-Rendering-Studio-neural-v1`. Written 2026-10-17. It replaces the 2026-10-16 handoff;
that work is done and summarised in §5. Agent rules are in `CLAUDE.md`.

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
- ☐ Kaggle spec `configs/kaggle/cad_dataset.json`: install core `@737dc35[occt,mesh]` inside the kernel.

### K1 — Dataset `berkelium-cad-v1` (built and verified **on Kaggle**)
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
- ☐ **D-033 (write before training).** M2 (Qwen3-8B + LoRA) is promoted to the Studio `cad` generator only if,
  on held-out data:
  - in-distribution test: schema-valid ≥ 95 %, realise (L3) ≥ 90 %, requirement pass ≥ 85 %;
  - held-out families: realise ≥ 60 %, and mean voxel IoU against the reference beats zero-shot Qwen3-8B
    (paired 95 % CI above 0);
  - gear test (core `berkelium eval` metrics): not worse than zero-shot Qwen3-8B;
  - 0 proposals that contain measurement or validation fields (schema `extra=forbid` rejects them).
- ☐ `configs/kaggle/cad_train_qwen3_8b.json`. QLoRA: NF4 base, fp16 compute (the T4 has no bf16), LoRA r=32 on
  all linear layers, seq ≤ 3072, gradient checkpointing, checkpoint every 30 min, resume across 12 h sessions,
  NaN guard. Compact JSON targets (sorted keys, no spaces).
- ☐ Adapter goes to HF `Mhaquehaque/berkelium-qwen3-8b-cad-v1` with a model card + run manifest (data/config
  SHA-256).

### K3 — Evaluate on Kaggle
- ☐ `configs/kaggle/cad_eval.json`: vLLM, fp16, tensor-parallel across 2×T4, LoRA (fallback: HF 4-bit batched).
  Generate for zero-shot Qwen3-8B and M2 on test + held-out families + gear test.
- ☐ Score inside the kernel with the core: parse → schema → realise → requirements → voxel IoU. Repair loop
  (≤ 3 JSON-Patch rounds) is reported separately, as "with repair".
- ☐ `results/cad/E-C1_*.json` + `.report.md`. Apply D-033 and record the decision in `NEURAL_RENDERING_DECISIONS.md`.

### K4 — Studio integration
- ☐ `cad` generator: text → M2 (Kaggle/HF endpoint or local GPU) → core `run()` → GLB → mesh → Gaussians → render.
  `open_questions` are shown to the user instead of guessing. The provenance card shows the proposal, the
  validation report, artifact hashes and the adapter hash.
- ☐ Router: an engineering intent goes to `cad`; organic objects go to `auto` open-world, labelled.
- ☐ Tests use a stub model (no GPU in CI). Docs: README, ROADMAP, ENGINEERING_LOG, THIRD_PARTY_NOTICES (Qwen3
  Apache-2.0; OCCT LGPL; core license still undecided, U8).

### Budget (Kaggle T4 hours, estimates)
K1 build ≈ 1–2 h (CPU-heavy, run inside a Kaggle session) · K2 train ≈ 6–10 h (two sessions) · K3 eval ≈ 2–3 h.
Total ≈ 10–15 h of the weekly ~30 h.

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
