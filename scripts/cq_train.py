#!/usr/bin/env python3
"""LoRA fine-tune of Qwen3 as the universal text → CadQuery compiler (E-C2), on one MI300X (ROCm, bf16).

  python scripts/cq_train.py --data ~/berkelium_run/data/cq_v1 --out ~/berkelium_run/train
  python scripts/cq_train.py --smoke --data ... --out ...       # Qwen3-0.6B, 20 steps

Recipe: Unsloth ``FastLanguageModel`` (falls back to plain transformers + PEFT if Unsloth cannot load on this
ROCm stack; the reason is recorded), bf16 weights and compute, LoRA r=64 / alpha=64 on every linear projection
(q,k,v,o,gate,up,down), sequence 2048 with greedy packing, 1 epoch, lr 1e-4 cosine (3 % warm-up), thinking off
(Qwen3 template with ``enable_thinking=False``). Loss is on the assistant completion only.

Size rule (recorded in ``<out>/size_decision.json``): start on Qwen/Qwen3-32B; after 200 optimizer steps measure
tokens/s; if the projected epoch is > 10 h, stop and restart on Qwen/Qwen3-14B with all the data.

Safety: a checkpoint (adapter + optimizer + scheduler + RNG) every 30 min; a re-run auto-resumes from the last
one; ``--time-budget-h`` stops cleanly with a checkpoint; a NaN/inf loss stops the run, is recorded in
``status.json`` and exits 3 (the bad state is not checkpointed). Promotion is decided only by scripts/cq_eval.py
under D-034."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "cadquery")]

import cqformat  # noqa: E402
import pins  # noqa: E402

BIG, MID, SMOKE = "Qwen/Qwen3-32B", "Qwen/Qwen3-14B", "Qwen/Qwen3-0.6B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PROBE_STEPS = 200
MAX_EPOCH_H = 10.0


def size_tag(model: str) -> str:
    return model.split("-")[-1].lower()          # "32b" / "14b" / "0.6b"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_rows(p: Path) -> list[dict]:
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def example_messages(row: dict) -> list[dict]:
    if row.get("kind") == "repair":
        return cqformat.repair_messages(row["text"], row["broken_completion"], row["error"])
    return cqformat.messages(row["text"])


def encode(tok, row: dict, max_len: int) -> dict | None:
    prompt = tok.apply_chat_template(example_messages(row), tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(row["completion"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    ids = p_ids + c_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + c_ids}


def pack(examples: list[dict], max_len: int, pad_id: int, seed: int) -> list[dict]:
    """Greedy first-fit packing into rows of ``max_len``; position_ids restart per example."""
    ex = list(examples)
    random.Random(seed).shuffle(ex)
    rows, cur = [], {"input_ids": [], "labels": [], "position_ids": []}
    for e in ex:
        if len(cur["input_ids"]) + len(e["input_ids"]) > max_len:
            rows.append(cur)
            cur = {"input_ids": [], "labels": [], "position_ids": []}
        cur["input_ids"] += e["input_ids"]
        cur["labels"] += e["labels"]
        cur["position_ids"] += list(range(len(e["input_ids"])))
    if cur["input_ids"]:
        rows.append(cur)
    for r in rows:
        r["n_tokens"] = len(r["input_ids"])
        pad = max_len - r["n_tokens"]
        r["input_ids"] += [pad_id] * pad
        r["labels"] += [-100] * pad
        r["position_ids"] += [0] * pad
        r["attention_mask"] = [1] * r["n_tokens"] + [0] * pad
    return rows


def load_model(model_id: str, revision: str, max_len: int, r: int, prefer_unsloth: bool):
    import torch
    why = None
    if prefer_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(model_name=model_id, revision=revision,
                                                           max_seq_length=max_len, dtype=torch.bfloat16,
                                                           load_in_4bit=False, full_finetuning=False)
            model = FastLanguageModel.get_peft_model(model, r=r, lora_alpha=r, lora_dropout=0.0, bias="none",
                                                     target_modules=TARGETS, use_gradient_checkpointing="unsloth",
                                                     random_state=0)
            return model, tok, "unsloth", None
        except Exception as e:  # noqa: BLE001 — any Unsloth/ROCm failure falls back, with the reason recorded
            why = f"{type(e).__name__}: {e}"[:800]
            print("Unsloth unavailable, falling back to transformers+PEFT:", why, flush=True)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=r, lora_dropout=0.0, target_modules=TARGETS,
                                             task_type="CAUSAL_LM"))
    return model, tok, "transformers+peft", why


def last_checkpoint(d: Path) -> str | None:
    cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1)
    cks = [c for c in cks if (c / "trainer_state.json").exists()]
    return str(cks[-1]) if cks else None


def train(a, model_id: str) -> int:
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments
    out_root = Path(a.out).expanduser()
    run_dir = out_root / f"qwen3-{size_tag(model_id)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_p = run_dir / "status.json"
    rev = pins.MODELS[model_id]["revision"]
    data = Path(a.data).expanduser()
    t_start = time.time()
    status = {"model": model_id, "revision": rev, "state": "starting", "started_unix": int(t_start)}

    def save_status(**kw):
        status.update(kw)
        status_p.write_text(json.dumps(status, indent=1, default=str))
    if (run_dir / "adapter" / "adapter_config.json").exists() and status_p.exists() and not a.smoke:
        if json.loads(status_p.read_text() or "{}").get("state") == "done":
            print("already trained:", run_dir / "adapter")
            return 0
    save_status()

    model, tok, backend, fallback_reason = load_model(model_id, rev, a.max_len, a.lora_r, not a.no_unsloth)
    save_status(backend=backend, unsloth_fallback_reason=fallback_reason)
    rows = load_rows(data / "train.jsonl.gz")
    if a.smoke:
        rows = rows[:200]
    enc, dropped = [], 0
    for r in rows:
        e = encode(tok, r, a.max_len)
        if e is None:
            dropped += 1
        else:
            enc.append(e)
    packed = pack(enc, a.max_len, tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id, a.seed)
    tokens_total = sum(r["n_tokens"] for r in packed)
    print(f"{len(rows)} rows -> {len(enc)} examples ({dropped} > {a.max_len} tokens dropped) -> {len(packed)} packed "
          f"rows, {tokens_total} tokens", flush=True)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(packed)

        def __getitem__(self, i):
            r = packed[i]
            return {k: torch.tensor(r[k]) for k in ("input_ids", "labels", "attention_mask", "position_ids")}

    world_bs = a.batch * a.grad_accum
    steps_epoch = math.ceil(len(packed) / world_bs)
    max_steps = a.max_steps or steps_epoch
    tok_per_step = tokens_total / max(1, steps_epoch)
    log: list[dict] = []
    manifest = {"experiment": "E-C2", "decision": "D-034", "model": model_id, "revision": rev, "backend": backend,
                "unsloth_fallback_reason": fallback_reason,
                "data": {"dir": str(data), "train_sha256": sha256_file(data / "train.jsonl.gz"),
                         "manifest_sha256": sha256_file(data / "manifest.json") if (data / "manifest.json").exists() else None,
                         "rows": len(rows), "examples": len(enc), "dropped_too_long": dropped,
                         "packed_rows": len(packed), "tokens": tokens_total},
                "config": {"lora_r": a.lora_r, "lora_alpha": a.lora_r, "targets": TARGETS, "max_len": a.max_len,
                           "lr": a.lr, "schedule": "cosine", "warmup_ratio": 0.03, "epochs": 1, "batch": a.batch,
                           "grad_accum": a.grad_accum, "dtype": "bf16", "thinking": False, "seed": a.seed,
                           "packing": "greedy first-fit, position_ids reset per example (cross-example attention is "
                                      "masked only by kernels that honour position_ids, e.g. flash-attention varlen)",
                           "system_prompt": cqformat.SYSTEM, "format": cqformat.FORMAT_VERSION},
                "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    class Guard(TrainerCallback):
        def __init__(self):
            self.last_save = time.time()
            self.t_train0 = None
            self.step0 = None

        def on_step_end(self, args, state, control, **kw):
            now = time.time()
            if self.t_train0 is None:
                self.t_train0, self.step0 = now, state.global_step
            done = state.global_step - self.step0
            if (not a.smoke and model_id == BIG and not a.no_size_switch and done == PROBE_STEPS):
                sec_step = (now - self.t_train0) / done
                proj_h = sec_step * (max_steps - state.global_step) / 3600 + (now - t_start) / 3600
                rec = {"probe_model": model_id, "probe_steps": done, "seconds_per_step": round(sec_step, 3),
                       "tokens_per_s": round(tok_per_step / sec_step, 1), "steps_per_epoch": steps_epoch,
                       "projected_epoch_h": round(proj_h, 2), "limit_h": MAX_EPOCH_H,
                       "decision": "switch to Qwen/Qwen3-14B" if proj_h > MAX_EPOCH_H else "keep Qwen/Qwen3-32B",
                       "measured_unix": int(now)}
                (out_root / "size_decision.json").write_text(json.dumps(rec, indent=1))
                print("size probe:", rec, flush=True)
                if proj_h > MAX_EPOCH_H:
                    save_status(state="switching", size_probe=rec)
                    control.should_training_stop = True
                    return control
            if now - self.last_save > a.ckpt_minutes * 60:
                control.should_save = True
                self.last_save = now
            if a.time_budget_h and now - t_start > a.time_budget_h * 3600:
                save_status(state="time_budget", stop_reason=f"time budget {a.time_budget_h} h at step {state.global_step}")
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            log.append({"step": state.global_step, **logs, "elapsed_h": round((time.time() - t_start) / 3600, 3)})
            with open(run_dir / "train_log.jsonl", "a") as f:
                f.write(json.dumps(log[-1]) + "\n")
            loss = logs.get("loss")
            if loss is not None and not math.isfinite(loss):
                save_status(state="nan", stop_reason=f"non-finite loss {loss} at step {state.global_step}")
                control.should_training_stop = True
                control.should_save = False
            return control

    targs = TrainingArguments(
        output_dir=str(run_dir), per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, num_train_epochs=1, max_steps=max_steps,
        bf16=True, logging_steps=1 if a.smoke else 10, save_strategy="no", save_total_limit=2, report_to=[],
        seed=a.seed, dataloader_num_workers=2, remove_unused_columns=False, optim="adamw_torch_fused",
        weight_decay=0.0, max_grad_norm=1.0, gradient_checkpointing=backend != "unsloth")
    trainer = Trainer(model=model, args=targs, train_dataset=DS(), callbacks=[Guard()])
    resume = last_checkpoint(run_dir)
    save_status(state="training", resume_from=resume, steps_per_epoch=steps_epoch, max_steps=max_steps)
    print("resume from:", resume, flush=True)
    res = trainer.train(resume_from_checkpoint=resume)
    st = json.loads(status_p.read_text())
    if st.get("state") == "nan":
        return 3
    if st.get("state") == "switching":
        return 10
    if trainer.state.global_step < max_steps:
        trainer._save_checkpoint(trainer.model, trial=None)
        save_status(state="stopped", step=trainer.state.global_step)
        return 4
    model.save_pretrained(str(run_dir / "adapter"))
    tok.save_pretrained(str(run_dir / "adapter"))
    adapter_hash = {p.name: sha256_file(p) for p in (run_dir / "adapter").iterdir() if p.is_file()}
    manifest.update(adapter_sha256=adapter_hash, train_loss=res.training_loss, steps=trainer.state.global_step,
                    hours=round((time.time() - t_start) / 3600, 3), log_tail=log[-5:])
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    save_status(state="done", adapter=str(run_dir / "adapter"), steps=trainer.state.global_step,
                train_loss=res.training_loss)
    (out_root / "final.json").write_text(json.dumps({"model": model_id, "revision": rev,
                                                     "adapter": str(run_dir / "adapter")}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="~/berkelium_run/data/cq_v1")
    ap.add_argument("--out", default="~/berkelium_run/train")
    ap.add_argument("--model", default=None, help="force a model (default: the size rule)")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--ckpt-minutes", type=float, default=30)
    ap.add_argument("--time-budget-h", type=float, default=0, help="0 = no limit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-unsloth", action="store_true")
    ap.add_argument("--no-size-switch", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.max_steps = a.max_steps or 20
        return train(a, a.model or SMOKE)
    model = a.model
    if model is None:
        dec = out / "size_decision.json"
        model = MID if dec.exists() and json.loads(dec.read_text())["decision"].endswith("14B") else BIG
    rc = train(a, model)
    if rc == 10:                                   # size rule fired: fresh process so the 32B weights are freed
        print("restarting on Qwen/Qwen3-14B (see size_decision.json)", flush=True)
        os.execv(sys.executable, [sys.executable, __file__, *sys.argv[1:], "--model", MID])
    return rc


if __name__ == "__main__":
    sys.exit(main())
