"""Chat-completion backends for the text → CadQuery compiler: vLLM (ROCm or CUDA), HF transformers, or an
OpenAI-compatible endpoint (vLLM server / HF Inference Endpoint). Greedy decoding, thinking off, as trained.

Every backend exposes ``chat(batch_of_messages, adapter: bool) -> list[str]``; ``adapter=False`` gives the
zero-shot base model from the same weights, so base and fine-tuned are compared under identical decoding."""

from __future__ import annotations

import os

MAX_NEW = 2048


class VLLMBackend:
    name = "vllm"

    def __init__(self, base: str, revision: str, adapter: str | None, max_len: int = 4096, tp: int = 1):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=base, revision=revision, dtype="bfloat16", max_model_len=max_len,
                       tensor_parallel_size=tp, enable_lora=adapter is not None, max_lora_rank=64,
                       gpu_memory_utilization=0.90, seed=0)
        self.sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
        self.adapter = adapter

    def chat(self, batch: list[list[dict]], adapter: bool = True) -> list[str]:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("cq", 1, self.adapter) if adapter and self.adapter else None
        res = self.llm.chat(batch, self.sp, lora_request=lora, use_tqdm=False,
                            chat_template_kwargs={"enable_thinking": False})
        return [r.outputs[0].text for r in res]


class HFBackend:
    name = "transformers"

    def __init__(self, base: str, revision: str, adapter: str | None, batch: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(base, revision=revision, padding_side="left")
        m = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16, device_map="auto")
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter)
        self.m, self.has_adapter, self.batch = m.eval(), bool(adapter), batch
        self.end = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id]

    def chat(self, batch: list[list[dict]], adapter: bool = True) -> list[str]:
        import contextlib

        import torch
        out = []
        ctx = contextlib.nullcontext() if adapter or not self.has_adapter else self.m.disable_adapter()
        with ctx, torch.no_grad():
            for i in range(0, len(batch), self.batch):
                ps = [self.tok.apply_chat_template(x, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                      for x in batch[i:i + self.batch]]
                enc = self.tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(self.m.device)
                g = self.m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=self.end,
                                    pad_token_id=self.tok.pad_token_id)
                out += [self.tok.decode(r[enc["input_ids"].shape[1]:], skip_special_tokens=True) for r in g]
        return out


class EndpointBackend:
    """OpenAI-compatible ``/chat/completions`` (``BERKELIUM_CAD_URL``; key from ``BERKELIUM_CAD_KEY`` or HF_TOKEN)."""
    name = "endpoint"

    def __init__(self, url: str, model: str, key: str | None = None, timeout: float = 300):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.key = key or os.environ.get("BERKELIUM_CAD_KEY") or os.environ.get("HF_TOKEN")

    def chat(self, batch: list[list[dict]], adapter: bool = True) -> list[str]:
        import httpx
        out = []
        for msgs in batch:
            r = httpx.post(self.url + "/chat/completions", timeout=self.timeout,
                           headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
                           json={"model": self.model, "messages": msgs, "temperature": 0, "max_tokens": MAX_NEW,
                                 "chat_template_kwargs": {"enable_thinking": False}})
            r.raise_for_status()
            out.append(r.json()["choices"][0]["message"]["content"])
        return out


def make_backend(base: str, revision: str, adapter: str | None, prefer: str = "vllm", tp: int = 1):
    """vLLM if importable and usable, else HF transformers. Returns (backend, fallback_reason)."""
    if prefer == "vllm":
        try:
            return VLLMBackend(base, revision, adapter, tp=tp), None
        except Exception as e:  # noqa: BLE001 — ImportError or a ROCm/vLLM init failure: fall back, recorded
            why = f"{type(e).__name__}: {e}"[:600]
            return HFBackend(base, revision, adapter), why
    return HFBackend(base, revision, adapter), None
