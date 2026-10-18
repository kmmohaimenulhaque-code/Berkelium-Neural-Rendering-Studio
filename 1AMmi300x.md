# MI300X run log: universal-cadquery (E-C2), 2026-10-17

Status of the owner's run on the droplet `Berkelium-gpu-mi300x1-192gb-devcloud-atl1`. The numbers come from the run
log the owner pasted (`~/berkelium_run/logs/`, on the droplet, not in git). **No training or eval number exists yet.**
The rule that decides promotion is D-034 (`NEURAL_RENDERING_DECISIONS.md`, commit `ff8a982`).

## 1. Host (as reported)
- 1× AMD Instinct MI300X VF in `rocm-smi`. The login banner says "5x MI300X"; only GPU[0] is listed, and the run uses one GPU.
- ROCm 7.14 driver. Python env: **Unsloth Studio's env** `/root/.unsloth/studio/unsloth_studio` (Python 3.13),
  symlinked as `~/berkelium_run/venv`.
- Versions frozen by `~/berkelium_run/constraints.txt`: torch 2.11.0+rocm7.2 (hip 7.2.26015), unsloth 2026.9.11,
  unsloth-zoo 2026.9.7, transformers 5.5.0, peft 0.18.1, accelerate 1.15.0, numpy 2.5.2. CadQuery 2.8.0 and the
  core `737dc35` are installed on top of these.
- **vLLM is not installed.** Text generation and eval use HF transformers instead, which is slower; the switch is recorded.
- 607 GB free at start. `HF_TOKEN` comes from the shell only. **The first token was pasted into chat. Revoke it.**

## 2. What happened (UTC, 2026-10-17)
| time | event |
|---|---|
| 13:43 | 1st launch: MI300X check failed (false negative: `grep -q` + `pipefail` SIGPIPE). Fixed on the branch |
| 14:08 | no torch in system python. Fixed: the script now installs the ROCm torch wheel when torch is missing |
| 14:15 | Unsloth missing from that venv. Ran with the PEFT fallback |
| 14:24 | smoke training failed: transformers 5 removed `warmup_ratio`. Fixed (`warmup_steps` = 3 % of steps) |
| 14:27 | **smoke passed** (58 tests; Qwen3-0.6B 20 steps, loss 2.54 → 1.04, no NaN; that run used transformers+PEFT) |
| 14:27–15:2x | data stage (1st try): text2cadquery 107,785 / 110,263 verified; families 19,653 / 20,026 |
| 15:2x | stopped to switch to Unsloth's env. A nested venv hid its torch, and the pip install pulled a non-ROCm torch. Fixed: venv = symlink to Unsloth's env, and the constraints now read installed versions |
| 15:32 | env stage OK on Unsloth's env; smoke skipped (already done); **data stage restarted** |
| ~15:43 | text2cadquery 107,786 / 110,263 verified (624.7 s) |
| ~15:56 | families 19,647 / 20,026 verified (792.6 s) |
| then | CAD-Recode clone + sandbox verification, then Qwen3-32B loaded (707 weight files) for request-text generation (HF transformers, cap `GEN_HOURS=2`). rocm-smi: GPU 100 %, 739 W / 750 W, VRAM 42 %, 69 °C |

Open item: smoke has **not** been re-run on the Unsloth env. If `=== train ===` fails, this is the first suspect.
Manual check (a second terminal is fine while data runs):
```bash
cd ~/berkelium_run/repo
~/berkelium_run/venv/bin/python scripts/cq_train.py --smoke --data ~/berkelium_run/data/smoke --out ~/berkelium_run/train_smoke_unsloth 2>&1 | tail -25
```

## 3. What comes next (automatic)
1. `cad_recode: {...}` line once the 2 h text cap is reached (only scripts that got a text are used), then `repair: {...}`,
   dedup, and the data files + `manifest.json` in `~/berkelium_run/data/cq_v1/`.
2. `d034` gate: refuses to train unless D-034 is committed in the branch history.
3. `train`: Qwen3-32B bf16 + LoRA r=64 on all linear layers, seq 2048 packed, 1 epoch, lr 1e-4 cosine, checkpoint every
   30 min, auto-resume. After 200 steps it writes `~/berkelium_run/size_decision.json`; if the projected epoch is
   > 10 h it restarts on Qwen3-14B with the same data.
4. `eval`: base vs fine-tuned on held-out data → `results/cadquery/E-C2.json` + `E-C2.report.md`, D-034 PASS/FAIL.
5. `publish`: private HF repos `Mhaquehaque/berkelium-qwen3-<32b|14b>-cadquery-v1` (adapter) and `...-merged`
   (full bf16). Results are committed on the droplet + `~/berkelium_run/results.bundle`. **No GitHub push** unless
   `BERKELIUM_GIT_PUSH=1`.

## 4. Monitor
```bash
tail -f ~/berkelium_run/logs/latest.log
```
```bash
ps -p $(cat ~/berkelium_run/run.pid) >/dev/null && echo RUN-ALIVE || echo RUN-DEAD
ls ~/berkelium_run/markers/
```
```bash
watch -n 2 'rocm-smi --showuse --showmemuse --showpower --showtemp'
```
```bash
watch -n 10 'tail -n 3 ~/berkelium_run/logs/latest.log; echo; rocm-smi --showuse --showmemuse --showpower | grep -E "^0|GPU"; echo; ls ~/berkelium_run/markers/'
```
Live Python stack (shows `generate` while texts are being written):
```bash
~/berkelium_run/venv/bin/python -m pip install -q py-spy
PID=$(pgrep -f "cq_data.py build" | head -1); echo "cq_data pid $PID"
~/berkelium_run/venv/bin/py-spy dump --pid $PID | head -30
```
Once training starts:
```bash
cat ~/berkelium_run/size_decision.json
cat ~/berkelium_run/train/qwen3-*/status.json
tail -n 3 ~/berkelium_run/train/qwen3-*/train_log.jsonl
```

## 5. Stop / resume (full commands)
Stop:
```bash
kill $(cat ~/berkelium_run/run.pid)
```
Resume (finished stages are skipped; training resumes from its last checkpoint):
```bash
UPY=/root/.unsloth/studio/unsloth_studio/bin/python
export HF_TOKEN=<your token>
unset BERKELIUM_ALLOW_NO_UNSLOTH
cd ~/berkelium_run/repo && git pull -q
PATH=$(dirname $UPY):$PATH bash scripts/mi300x_oneshot.sh
tail -f ~/berkelium_run/logs/latest.log
```

## 6. After it finishes
```bash
cat ~/berkelium_run/repo/results/cadquery/E-C2.report.md
cat ~/berkelium_run/repo/results/cadquery/publish_receipt.json
cd ~/berkelium_run/repo && git push origin universal-cadquery
```
If the push from the droplet fails (no GitHub login), copy `~/berkelium_run/results.bundle` to the Codespace and run:
```bash
git fetch results.bundle universal-cadquery:tmp && git merge tmp && git push
```

## 7. Update (19:13Z → ~19:45Z): data done, training on 32B kept by owner decision
- `data done in 220 min` (19:13:48Z). `d034` gate passed: D-034 commit `ff8a9825f4e9960e3b93a12e1af57f759d6339c1`
  (2026-10-17T12:51:23Z), before training.
- Training input (from the train log): 121,575 train rows → 119,107 examples (2,468 longer than 2048 tokens
  dropped) → 44,470 packed rows, 74,800,876 tokens; batch 4 × grad-accum 4 = 16; **2,780 steps** for one epoch.
- Qwen3-32B + Unsloth 2026.9.11 (64 layers patched); trainable 536,870,912 of 33,298,994,176 parameters (1.61 %).
  Speed at step 47: 15.71 s/step, so the projected epoch is about 12.1 h.
- That projection is over the 10 h size rule (in `cq_train.py`, the plan's rule, not part of D-034), so the run would
  have switched to 14B at step 200. **The owner overrode it to keep 32B** (deadline re-checked). The run was
  restarted with `TRAIN_ARGS=--no-size-switch` before step 200 and before any checkpoint, losing about 12 min. This
  override is the reason no `size_decision.json` exists.
- The first 32B launch logged `Unsloth: Ignoring revision 9216db5… since Qwen/Qwen3-32B resolved to
  unsloth/Qwen3-32B`. Fixed on the branch: `use_exact_model_name=True` keeps the pinned Qwen revision. The
  restart uses the fix.
- Eval capped at `--per-source 200` (HF transformers, no vLLM). Publish is public (`--public`), as the teammate asked.
  GGUF Q4_K_M + teammate hand-off: `scripts/cq_gguf.sh` (waits for publish).
- Expected (estimate): training ~12.1 h + load, then eval and publish. No E-C2 number yet.

Restart command used:
```bash
kill $(cat ~/berkelium_run/run.pid) 2>/dev/null
pkill -f mi300x_oneshot.sh; pkill -f cq_train.py; sleep 10
rm -f ~/berkelium_run/size_decision.json
rm -rf ~/berkelium_run/train/qwen3-32b ~/berkelium_run/train/qwen3-14b
UPY=/root/.unsloth/studio/unsloth_studio/bin/python
export HF_TOKEN=<your token>
unset BERKELIUM_ALLOW_NO_UNSLOTH
export TRAIN_ARGS="--no-size-switch"
export EVAL_ARGS="--per-source 200"
export PUBLISH_ARGS="--public"
cd ~/berkelium_run/repo && git pull -q
PATH=$(dirname $UPY):$PATH bash scripts/mi300x_oneshot.sh
```
Monitor training:
```bash
tail -f ~/berkelium_run/logs/latest.log
cat ~/berkelium_run/train/qwen3-32b/status.json
tail -n 3 ~/berkelium_run/train/qwen3-32b/train_log.jsonl
ls ~/berkelium_run/train/qwen3-32b/ | grep checkpoint
```

## 8. Eval skipped (owner decision, deadline)
The E-C2 eval stage is skipped (marker `eval.done` written by hand with `skipped by owner`). **D-034 is not applied**:
the model is published as **opt-in, not evaluated**, and the model cards say so. No E-C2 number exists. Publish and
GGUF run without `E-C2.json`.

## 9. GGUF path tested on the droplet (22:00Z, CPU, while training ran)
Smoke Qwen3-0.6B + its 20-step LoRA, merged on CPU → `convert_hf_to_gguf.py` (llama.cpp `620d222`) → BF16 GGUF
1.2 GB (1137.00 MiB, 16.00 BPW) → `llama-quantize` Q4_K_M 379 MB (372.65 MiB, 5.24 BPW) in 3.1 s. The convert +
quantize path works for the Qwen3 architecture. Not tested yet: text generation from the GGUF (the `llama-cli` line
printed nothing) and Ollama reading the repo's `system`/`params`. `cq_gguf.sh` reuses this llama.cpp build.

## 10. Finished (2026-10-18)
- Training: Qwen3-32B, all 2,780 / 2,780 steps (one full epoch).
- Publish: the first attempt failed only at the final `git commit` (exit 128, no git identity on the droplet). The uploads
  had already finished. Fixed by hand (git identity set, `publish.done` written).
- GGUF: Q4_K_M 18,840.96 MiB (4.82 BPW) from the 62,490.08 MiB BF16; quantize 128 s; uploaded 08:24Z.
- Public repos and sizes: `results/cadquery/HANDOFF_TEAMMATE.md`. **Not evaluated** (D-034 not applied; opt-in).
