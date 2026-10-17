#!/usr/bin/env bash
# Berkelium universal text→CadQuery: one-shot data → train → eval → publish on an AMD MI300X (ROCm) host.
#
# Usage:   export HF_TOKEN=...; bash mi300x_oneshot.sh          (detaches itself; prints the `tail -f` command)
# Hours:   ESTIMATES, not measurements: checks+install ~0.5 h · smoke ~0.5 h · data 3–6 h (downloads, ~1.2M sandbox
#          runs, --gen-hours of vLLM text generation, default 2) · train: 32B probe ~1 h, then 32B or 14B for one
#          epoch (the script measures tokens/s after 200 steps; > 10 h projected → 14B) · eval 2–4 h · publish ~1 h.
#          Plan for ~1–2 days wall clock.
# Resume:  run the same command again. Finished stages are skipped (markers in ~/berkelium_run/markers/);
#          training resumes from its last 30-min checkpoint. `rm ~/berkelium_run/markers/<stage>.done` redoes one.
# Stop:    kill "$(cat ~/berkelium_run/run.pid)"     (or: pkill -f mi300x_oneshot.sh; then re-run to resume)
# Knobs:   GEN_HOURS=2  RECODE_MAX=1000000  BRANCH=universal-cadquery  RUN=~/berkelium_run
#          BERKELIUM_ALLOW_NO_UNSLOTH=1 (train with transformers+PEFT if Unsloth is missing)  BERKELIUM_GIT_PUSH=1
set -euo pipefail

RUN="${RUN:-$HOME/berkelium_run}"
BRANCH="${BRANCH:-universal-cadquery}"
REPO_URL="${REPO_URL:-https://github.com/enigmahacker/Berkelium-Neural-Rendering-Studio.git}"
CORE_PIN="berkelium[occt,mesh] @ git+https://github.com/kmmohaimenulhaque-code/BERKELIUM-NEURAL-RENDERING-AND-SCIENCES@737dc35c42aaf97f52a6ac984a806adef9518525"
GEN_HOURS="${GEN_HOURS:-2}"
RECODE_MAX="${RECODE_MAX:-1000000}"
LOGS="$RUN/logs"; MARK="$RUN/markers"; SRC="$RUN/repo"; VENV="$RUN/venv"
mkdir -p "$LOGS" "$MARK" "$RUN/data" "$RUN/train" "$RUN/eval" "$RUN/cache"

# ---------------------------------------------------------------- detach (nohup + setsid; env incl. HF_TOKEN is inherited, never written)
if [[ "${BERKELIUM_DETACHED:-0}" != 1 ]]; then
  if [[ -f "$RUN/run.pid" ]] && kill -0 "$(cat "$RUN/run.pid")" 2>/dev/null; then
    echo "already running (pid $(cat "$RUN/run.pid")). Watch: tail -f $LOGS/latest.log"; exit 0; fi
  [[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: HF_TOKEN is not set. Run: export HF_TOKEN=<your token>   (it is never printed or stored)"; exit 2; }
  SELF="$RUN/oneshot.sh"; [[ "$(readlink -f "$0")" == "$(readlink -f "$SELF")" ]] || cp "$0" "$SELF"
  LOG="$LOGS/oneshot_$(date -u +%Y%m%dT%H%M%SZ).log"; ln -sfn "$LOG" "$LOGS/latest.log"
  BERKELIUM_DETACHED=1 nohup setsid bash "$SELF" >"$LOG" 2>&1 < /dev/null &
  echo $! > "$RUN/run.pid"
  echo "started (pid $(cat "$RUN/run.pid")). Watch:  tail -f $LOGS/latest.log"
  echo "stop:  kill \$(cat $RUN/run.pid)      resume: run the same command again"
  exit 0
fi

ts() { date -u +%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }
die() { echo "[$(ts)] FAILED: $*"; echo "fix the cause and re-run; finished stages are skipped"; exit 1; }
stage() {  # stage <name> <command...>: skip if marker exists, log to logs/<name>.log
  local name=$1; shift
  if [[ -f "$MARK/$name.done" ]]; then log "skip $name (done $(cat "$MARK/$name.done"))"; return 0; fi
  log "=== $name ==="; local t0=$SECONDS rc
  set +e; ( set -e; "$@" ) 2>&1 | tee -a "$LOGS/$name.log"; rc=${PIPESTATUS[0]}; set -e
  (( rc == 0 )) || die "$name exited $rc (see $LOGS/$name.log)"
  date -u +%FT%TZ > "$MARK/$name.done"; log "$name done in $(( (SECONDS - t0) / 60 )) min"
}
trap 'rc=$?; echo "[$(ts)] exited with status $rc"' EXIT
echo $$ > "$RUN/run.pid"
log "Berkelium one-shot on $(hostname); run dir $RUN; branch $BRANCH"

# ---------------------------------------------------------------- checks
command -v rocm-smi >/dev/null || die "rocm-smi not found: this is not a ROCm host"
PRODUCT=$(rocm-smi --showproductname 2>&1 || true)   # capture first: `| grep -q` + pipefail fails on SIGPIPE
[[ "${PRODUCT^^}" == *MI300* ]] || die "rocm-smi does not list an MI300X: $(echo "$PRODUCT" | head -8)"
log "ok: rocm-smi sees an MI300X"
PY=python3; command -v "$PY" >/dev/null || die "python3 not found"
if ! "$PY" -c "import torch" 2>/dev/null; then
  # bare ROCm host (no PyTorch image): put the official ROCm torch wheel in the venv instead
  log "no torch in system python: installing ROCm PyTorch (${TORCH_INDEX:=https://download.pytorch.org/whl/rocm6.4}) into $VENV"
  if ! "$PY" -m venv --help >/dev/null 2>&1 || ! "$PY" -c "import ensurepip" 2>/dev/null; then
    [[ $(id -u) == 0 ]] && apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip python3-dev git >/dev/null
  fi
  [[ -x "$VENV/bin/python" ]] || "$PY" -m venv "$VENV" || die "cannot create venv (apt-get install python3-venv)"
  "$VENV/bin/python" -m pip install -q -U pip
  "$VENV/bin/python" -m pip install -q torch --index-url "$TORCH_INDEX" || die "ROCm torch install failed"
  PY="$VENV/bin/python"
  "$PY" -c "import unsloth" 2>/dev/null || { export BERKELIUM_ALLOW_NO_UNSLOTH=1; log "no Unsloth on a bare host: training uses transformers+PEFT (recorded)"; }
fi
"$PY" - <<'EOF' || die "PyTorch ROCm check failed (needs the image's ROCm torch; torch.version.hip must be set)"
import torch, sys
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
assert torch.version.hip, f"torch {torch.__version__} is not a ROCm build"
print(f"ok: torch {torch.__version__} hip {torch.version.hip} device {torch.cuda.get_device_name(0)}")
EOF
if "$PY" -c "import unsloth" >/dev/null 2>&1; then log "ok: unsloth imports"
elif [[ "${BERKELIUM_ALLOW_NO_UNSLOTH:-0}" == 1 ]]; then log "WARNING: unsloth missing; training will use transformers+PEFT (recorded)"
else die "unsloth does not import from the image. Use an image with Unsloth for ROCm, or set BERKELIUM_ALLOW_NO_UNSLOTH=1 to train with transformers+PEFT"; fi
FREE_GB=$(df -BG --output=avail "$RUN" | tail -1 | tr -dc 0-9)
(( FREE_GB >= 500 )) || die "only ${FREE_GB} GB free under $RUN; need >= 500 GB"
log "ok: ${FREE_GB} GB free"
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is not set"
log "ok: HF_TOKEN is set (not shown)"

# ---------------------------------------------------------------- code + env
if [[ -d "$SRC/.git" ]]; then git -C "$SRC" fetch -q origin "$BRANCH" && git -C "$SRC" checkout -q "$BRANCH" && git -C "$SRC" pull -q --ff-only origin "$BRANCH"
else git clone -q -b "$BRANCH" "$REPO_URL" "$SRC"; fi
log "code: $(git -C "$SRC" rev-parse HEAD)"
cd "$SRC"
setup_env() {
  [[ -x "$VENV/bin/python" ]] || "$PY" -m venv --system-site-packages "$VENV"
  local P="$VENV/bin/python"
  # never replace the image's ROCm torch / vllm / unsloth / transformers stack: freeze what is there
  # (pip freeze prints ROCm wheels as "torch @ file://..." so read the installed versions directly)
  "$P" - > "$RUN/constraints.txt" <<'PYC'
import importlib.metadata as md
for n in ("torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm", "vllm", "unsloth", "unsloth-zoo",
          "transformers", "peft", "accelerate", "numpy", "bitsandbytes"):
    try:
        print(f"{n}=={md.version(n)}")
    except md.PackageNotFoundError:
        pass
PYC
  cat "$RUN/constraints.txt"
  local extra=()
  for m in peft transformers accelerate huggingface_hub; do "$P" -c "import $m" 2>/dev/null || extra+=("$m"); done
  "$P" -m pip install -q -c "$RUN/constraints.txt" "cadquery==2.8.0" "$CORE_PIN" httpx trimesh pytest "${extra[@]}"
  if ! "$P" -c "import OCP" 2>/dev/null; then
    if [[ $(id -u) == 0 ]] && command -v apt-get >/dev/null; then apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 libxrender1 libxext6 >/dev/null; fi
    "$P" -c "import OCP" || { echo "OCP (OpenCascade) does not import; install libGL (apt-get install libgl1)"; return 1; }
  fi
  "$P" -c "import cadquery, berkelium; print('cadquery', cadquery.__version__, '| core ok')"
  "$P" -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm not importable: eval/text generation will use HF transformers (recorded)"
  "$P" -c "import torch; assert torch.version.hip, 'venv torch is not ROCm'; print('venv torch', torch.__version__, torch.version.hip)"
}
stage env setup_env
P="$VENV/bin/python"

# ---------------------------------------------------------------- stages
smoke() {
  "$P" -m pytest cadquery/tests -q -x
  "$P" scripts/cq_data.py build --smoke --sources families,repair --out "$RUN/data/smoke" --cache "$RUN/cache" --gen-hours 0
  "$P" scripts/cq_train.py --smoke --data "$RUN/data/smoke" --out "$RUN/train_smoke"
}
stage smoke smoke

stage data "$P" scripts/cq_data.py build --out "$RUN/data/cq_v1" --cache "$RUN/cache" --gen-hours "$GEN_HOURS" --recode-max "$RECODE_MAX"

d034_check() {
  grep -q "| D-034 |" NEURAL_RENDERING_DECISIONS.md || { echo "D-034 is not in NEURAL_RENDERING_DECISIONS.md: refusing to train"; return 1; }
  local c; c=$(git log --format=%H --reverse -S "| D-034 |" -- NEURAL_RENDERING_DECISIONS.md | head -1)
  [[ -n "$c" ]] || { echo "D-034 is not committed: refusing to train"; return 1; }
  git merge-base --is-ancestor "$c" HEAD || { echo "D-034 commit $c is not in this branch's history"; return 1; }
  echo "D-034 pre-registered in commit $c ($(git log -1 --format=%cI "$c"))"
  echo "$c" > "$RUN/d034_commit.txt"
}
stage d034 d034_check

train() {
  local rc=0
  "$P" scripts/cq_train.py --data "$RUN/data/cq_v1" --out "$RUN/train" || rc=$?
  case $rc in
    0) ;;
    3) echo "training stopped on a NaN/inf loss (recorded in $RUN/train/*/status.json)"; return 3;;
    4) echo "training stopped before the end of the epoch (time budget); re-run to resume"; return 4;;
    *) return $rc;;
  esac
  [[ -f "$RUN/train/final.json" ]]
}
stage train train

evaluate() {
  local base adapter
  base=$("$P" -c "import json;print(json.load(open('$RUN/train/final.json'))['model'])")
  adapter=$("$P" -c "import json;print(json.load(open('$RUN/train/final.json'))['adapter'])")
  "$P" scripts/cq_eval.py --data "$RUN/data/cq_v1" --base "$base" --adapter "$adapter" --results results/cadquery --items-dir "$RUN/eval"
}
stage eval evaluate

stage publish "$P" scripts/cq_publish.py --train "$RUN/train" --results results/cadquery

# ---------------------------------------------------------------- summary
"$P" - "$RUN" <<'EOF'
import json, sys, pathlib
run = pathlib.Path(sys.argv[1]); r = json.loads(pathlib.Path("results/cadquery/E-C2.json").read_text())
p = json.loads(pathlib.Path("results/cadquery/publish_receipt.json").read_text())
size = run / "size_decision.json"
print("\n================ SUMMARY ================")
print("model      :", r["tuned"]["model"], "| adapter:", r["tuned"]["adapter"])
print("size rule  :", json.loads(size.read_text())["decision"] if size.exists() else "no probe recorded")
print("data       :", run / "data/cq_v1/manifest.json")
print("results    :", "results/cadquery/E-C2.json, results/cadquery/E-C2.report.md")
print("HF adapter : https://huggingface.co/" + p["adapter_repo"])
if p.get("merged_repo"): print("HF merged  : https://huggingface.co/" + p["merged_repo"])
for c in r["d034"]["criteria"]: print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['name']}: {c['value']} (need {c['threshold']})")
print("D-034      :", r["d034"]["verdict"], "->", r["d034"]["action"])
print("evidence   :", r["host"]["evidence"], "| sandbox net isolation:", r["sandbox"]["net_isolation"])
EOF
log "all stages finished"
