#!/usr/bin/env bash
# After the one-shot run: merged bf16 model → GGUF Q4_K_M (llama.cpp, pinned) → public HF repo <merged>-GGUF,
# plus the teammate hand-off (link, base model, file sizes, full weights vs LoRA). Weights never touch git.
#
# Usage:   export HF_TOKEN=...; bash scripts/cq_gguf.sh        (detaches; waits for the publish stage, then runs)
# Watch:   tail -f ~/berkelium_run/logs/gguf.log
# Output:  ~/berkelium_run/gguf/*.gguf, results/cadquery/HANDOFF_TEAMMATE.md (also printed at the end)
set -euo pipefail
RUN="${RUN:-$HOME/berkelium_run}"; SRC="$RUN/repo"; P="$RUN/venv/bin/python"; OUT="$RUN/gguf"
LLAMA_REF=620d2225cc436ab0b001e3200e590e3b286ef9ab     # ggml-org/llama.cpp release v0.7.0
QUANT="${QUANT:-Q4_K_M}"
mkdir -p "$RUN/logs" "$OUT"
if [[ "${GGUF_DETACHED:-0}" != 1 ]]; then
  [[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: export HF_TOKEN first"; exit 2; }
  GGUF_DETACHED=1 nohup setsid bash "$(readlink -f "$0")" > "$RUN/logs/gguf.log" 2>&1 < /dev/null &
  echo "started (pid $!). Watch: tail -f $RUN/logs/gguf.log"; exit 0
fi
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

log "waiting for the publish stage ($RUN/markers/publish.done)"
until [[ -f "$RUN/markers/publish.done" ]]; do sleep 60; done
cd "$SRC"
read -r BASE MERGED_REPO MERGED_DIR ADAPTER_REPO < <("$P" - <<'EOF'
import json, pathlib
r = json.loads(pathlib.Path("results/cadquery/publish_receipt.json").read_text())
size = r["base"].split("-")[-1].lower()
import os
print(r["base"], r.get("merged_repo", ""), os.path.expanduser(f"~/berkelium_run/train/merged-{size}"), r["adapter_repo"])
EOF
)
[[ -d "$MERGED_DIR" ]] || { log "merged model not found at $MERGED_DIR"; exit 1; }
GGUF_REPO="${MERGED_REPO%-merged}-GGUF"
NAME=$(basename "$GGUF_REPO" -GGUF)
log "base $BASE | merged $MERGED_DIR | target https://huggingface.co/$GGUF_REPO"

# ---- llama.cpp (CPU build is enough for convert + quantize)
if [[ ! -x "$RUN/llama.cpp/build/bin/llama-quantize" ]]; then
  [[ -d "$RUN/llama.cpp/.git" ]] || git clone -q https://github.com/ggml-org/llama.cpp "$RUN/llama.cpp"
  git -C "$RUN/llama.cpp" fetch -q origin "$LLAMA_REF" && git -C "$RUN/llama.cpp" checkout -q "$LLAMA_REF"
  command -v cmake >/dev/null || { apt-get update -qq && apt-get install -y -qq cmake build-essential >/dev/null; }
  cmake -S "$RUN/llama.cpp" -B "$RUN/llama.cpp/build" -DGGML_NATIVE=ON -DLLAMA_CURL=OFF >/dev/null
  cmake --build "$RUN/llama.cpp/build" --target llama-quantize -j "$(nproc)" >/dev/null
fi
"$P" -m pip install -q "gguf" sentencepiece protobuf
F16="$OUT/$NAME-BF16.gguf"; Q="$OUT/$NAME-$QUANT.gguf"
[[ -f "$F16" ]] || "$P" "$RUN/llama.cpp/convert_hf_to_gguf.py" "$MERGED_DIR" --outtype bf16 --outfile "$F16"
[[ -f "$Q" ]] || "$RUN/llama.cpp/build/bin/llama-quantize" "$F16" "$Q" "$QUANT"
log "quantized: $(du -h "$Q" | cut -f1) $Q"

# ---- Ollama extras read from an hf.co repo: system prompt (+ /no_think: thinking off as trained) and params
"$P" - "$OUT" <<'EOF'
import json, pathlib, sys
sys.path.insert(0, "cadquery"); import cqformat
out = pathlib.Path(sys.argv[1])
(out / "system").write_text(cqformat.SYSTEM + " /no_think")
(out / "params").write_text(json.dumps({"temperature": 0, "num_ctx": 4096, "stop": ["<|im_end|>"]}))
EOF

# ---- upload (public) + teammate hand-off
"$P" - "$GGUF_REPO" "$Q" "$OUT" "$BASE" "$MERGED_REPO" "$ADAPTER_REPO" "$QUANT" <<'EOF'
import hashlib, json, os, pathlib, sys
from huggingface_hub import HfApi
repo, q, out, base, merged, adapter, quant = sys.argv[1:]
q, out = pathlib.Path(q), pathlib.Path(out)
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, private=False, exist_ok=True)
for r in (adapter, merged):                      # the teammate asked for public repos
    if r:
        api.update_repo_settings(r, private=False)
ec2 = pathlib.Path("results/cadquery/E-C2.json")
res = json.loads(ec2.read_text()) if ec2.exists() else None
d = res["d034"] if res else {"verdict": "NOT EVALUATED", "action": "opt-in only; the E-C2 eval was skipped (deadline), D-034 not applied"}
final = json.loads(pathlib.Path(os.path.expanduser("~/berkelium_run/train/final.json")).read_text())
sha = hashlib.sha256(q.read_bytes()).hexdigest()
gb = q.stat().st_size / 1e9
card = f"""---
license: other
base_model: {base}
tags: [gguf, cadquery, cad, text-to-cad, berkelium, ollama]
---
# {repo.split('/')[-1]}

GGUF {quant} of [{merged}](https://huggingface.co/{merged}) (full weights: `{base}` + the E-C2 CadQuery LoRA, merged).

```
ollama run hf.co/{repo}:{quant}
```
The repo's `system` file sets the trained system prompt (thinking off). Output: one JSON header line, then one
```python CadQuery block ending `result = ...`. Run the script in the Berkelium sandbox, never as plain Python.

E-C2 / D-034 verdict: **{d['verdict']}** ({d['action']}). Quantization to {quant} has **not** been evaluated. Licence: trained partly on CAD-Recode
(CC-BY-NC-4.0) → **non-commercial**.
"""
(out / "README.md").write_text(card)
for f in (q, out / "system", out / "params", out / "README.md"):
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo)
size_merged = sum(p.stat().st_size for p in pathlib.Path(os.path.expanduser(
    f"~/berkelium_run/train/merged-{base.split('-')[-1].lower()}")).glob("*.safetensors")) / 1e9
size_adapter = sum(p.stat().st_size for p in pathlib.Path(final["adapter"]).glob("*.safetensors")) / 1e9
msg = f"""# BerkeliumGPT (local): hand-off

- **Ollama (GGUF {quant}, {gb:.1f} GB):** https://huggingface.co/{repo}  ->  `ollama run hf.co/{repo}:{quant}`
- **Full merged weights (bf16 safetensors, {size_merged:.1f} GB):** https://huggingface.co/{merged}
- **LoRA adapter only ({size_adapter:.2f} GB):** https://huggingface.co/{adapter}
- **Base model:** `{base}` @ `{final['revision']}`; trained {final.get('steps')} / {final.get('max_steps')} steps
- **Type:** the GGUF and `-merged` repos are full weights (base + LoRA merged); the adapter repo is LoRA only.
- **GGUF sha256:** `{sha}`
- **Eval:** D-034 **{d['verdict']}**: {d['action']}. The {quant} GGUF itself was not evaluated.
- **Use:** system prompt is in the repo (`system`); temperature 0; run the returned script in a sandbox.
- **Licence:** non-commercial (CAD-Recode CC-BY-NC-4.0 in the training data).
"""
pathlib.Path("results/cadquery/HANDOFF_TEAMMATE.md").write_text(msg)
r = json.loads(pathlib.Path("results/cadquery/publish_receipt.json").read_text())
r.update(gguf_repo=repo, gguf_file=q.name, gguf_sha256=sha, gguf_bytes=q.stat().st_size, public=True)
pathlib.Path("results/cadquery/publish_receipt.json").write_text(json.dumps(r, indent=1))
print(msg)
EOF
git add results/cadquery && git commit -qm "GGUF $QUANT published + teammate hand-off" || true
log "done. Hand-off: $SRC/results/cadquery/HANDOFF_TEAMMATE.md"
