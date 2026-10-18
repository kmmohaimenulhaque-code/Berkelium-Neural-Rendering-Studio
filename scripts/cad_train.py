#!/usr/bin/env python3
"""QLoRA fine-tune of Qwen3-8B as the Berkelium design compiler (E-C1 K2). Runs on Kaggle T4s (no bf16):
NF4 base weights, fp16 compute, fp32 LoRA weights, gradient checkpointing, DDP over both GPUs.

  torchrun --nproc_per_node 2 scripts/cad_train.py --data data/cad_v1 --out exports/qwen3-8b-cad-v1

Pre-registered rule: D-033 in NEURAL_RENDERING_DECISIONS.md (written before this run). This script only trains;
promotion is decided by scripts/cad_eval.py on held-out data.

Safety: a time budget stops training cleanly and saves the adapter before Kaggle's 12 h limit; a NaN/inf loss
stops the run and is recorded (never silently skipped)."""

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
sys.path.insert(0, str(ROOT / "services" / "neural"))

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().splitlines() if x.strip()]


def cem_catalogue() -> str:
    from berkelium.ai.prompts import cem_catalogue as cat
    from berkelium.cem.protocol import default_registry
    return cat(default_registry())


def encode(tok, row: dict, catalogue: str, max_len: int):
    from berkelium_neural.cad.prompts import compact, messages
    msgs = messages(row["text"], catalogue)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = prompt + compact(row["target"]) + "<|im_end|>\n"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(ids) > max_len or ids[:len(p_ids)] != p_ids:
        return None
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + ids[len(p_ids):]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cad_v1")
    ap.add_argument("--out", default="exports/qwen3-8b-cad-v1")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--time-budget-h", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Trainer, TrainerCallback, TrainingArguments)

    t_start = time.time()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    main_proc = rank == 0
    random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    tok.pad_token = tok.pad_token or "<|endoftext|>"
    cat = cem_catalogue()
    data = Path(a.data)
    train_rows = load_rows(data / "train.jsonl.gz")
    val_rows = load_rows(data / "val.jsonl.gz")[:120]
    random.shuffle(train_rows)
    if a.max_rows:
        train_rows = train_rows[:a.max_rows]
    enc = [encode(tok, r, cat, a.max_len) for r in train_rows]
    dropped = sum(e is None for e in enc)
    train = [e for e in enc if e is not None]
    val = [e for e in (encode(tok, r, cat, a.max_len) for r in val_rows) if e is not None]
    lens = sorted(len(e["input_ids"]) for e in train)
    stats = {"train_rows": len(train), "dropped_over_max_len": dropped, "val_rows": len(val),
             "tokens_total": sum(lens), "len_p50": lens[len(lens) // 2], "len_p95": lens[int(len(lens) * 0.95)],
             "len_max": lens[-1]}
    if main_proc:
        print(json.dumps(stats), flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, quantization_config=bnb,
                                                 dtype=torch.float16, device_map={"": rank},
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    # Not prepare_model_for_kbit_training: it upcasts every non-quantised weight to fp32, and Qwen3-8B's untied
    # 152k-vocab embedding + lm_head alone become ~5 GB per GPU -> OOM on a 15 GB T4 (seen 2026-10-17).
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    for n, p in model.named_parameters():          # fp32 adapters: fp16 LoRA weights underflow on T4 AMP
        if p.requires_grad:
            p.data = p.data.float()
    if main_proc:
        model.print_trainable_parameters()

    log: list[dict] = []
    status = {"stop_reason": "completed"}

    class Guard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                log.append({"step": state.global_step, **logs, "elapsed_h": round((time.time() - t_start) / 3600, 3)})
                lv = logs.get("loss")
                if lv is not None and not math.isfinite(lv):
                    status["stop_reason"] = f"non-finite loss {lv} at step {state.global_step}"
                    control.should_training_stop = True

        def on_step_end(self, args, state, control, **kw):
            if time.time() - t_start > a.time_budget_h * 3600:
                status["stop_reason"] = f"time budget {a.time_budget_h} h reached at step {state.global_step}"
                control.should_training_stop = True

    targs = TrainingArguments(
        output_dir=str(out / "trainer"), per_device_train_batch_size=a.bsz, per_device_eval_batch_size=a.bsz,
        gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.0, max_grad_norm=1.0, fp16=True,
        logging_steps=10, save_strategy="no", eval_strategy="no", report_to=[],
        ddp_find_unused_parameters=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, dataloader_num_workers=1, seed=a.seed,
        remove_unused_columns=False, optim="paged_adamw_8bit")
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[Guard()])
    res = trainer.train()
    ev = trainer.evaluate()
    if main_proc:
        adir = out / "adapter"
        for n, p in model.named_parameters():      # ship fp16 adapter weights (half the size; loads anywhere)
            if "lora_" in n:
                p.data = p.data.half()
        model.save_pretrained(str(adir), safe_serialization=True)
        tok.save_pretrained(str(adir))
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(adir.iterdir()) if p.is_file()}
        man = {"model": a.model, "revision": a.revision, "method": "QLoRA (NF4 base, fp16 compute, fp32 LoRA), DDP",
               "world_size": world, "config": vars(a), "data_manifest_sha256":
               json.loads((data / "manifest.json").read_text()).get("manifest_sha256"),
               "data_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob("*.gz"))},
               "stats": stats, "train_result": res.metrics, "eval": ev, "status": status,
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               "torch": torch.__version__, "hours": round((time.time() - t_start) / 3600, 3),
               "adapter_files_sha256": files, "log": log}
        (out / "run_manifest.json").write_text(json.dumps(man, indent=2, default=str))
        print(json.dumps({"status": status, "eval": ev, "hours": man["hours"]}), flush=True)
        if status["stop_reason"].startswith("non-finite"):
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
