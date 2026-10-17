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
