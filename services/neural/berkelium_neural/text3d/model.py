"""BerkeliumTextShape v1: text-conditioned implicit 3D generator (original Berkelium code, MIT).

  caption --tokenizer--> tokens --TextEncoder (transformer)--> z (128)
  (xyz, z) --SDFDecoder (Fourier features + MLP)--> signed distance + linear-RGB colour
  SDF on a grid --marching cubes--> triangle mesh (+ vertex colours) --> optional Gaussian scene

Fine-tuning mechanism actually supported by this model (and only this): LoRA on the nn.Linear layers listed in
``LORA_TARGETS`` plus trainable embedding deltas for named new tokens. The base weights stay frozen; an adapter is a
small separate state dict bound to the sha256 of the base checkpoint it was trained against.
"""

from __future__ import annotations

import math
import re

import torch
from torch import nn

from .shapes import COLORS, SHAPES, SIZES, TEMPLATES

PAD, UNK = "<pad>", "<unk>"
MAX_TOKENS = 16


def build_vocab() -> list[str]:
    words = set()
    for ts in TEMPLATES.values():
        for t in ts:
            words |= set(re.findall(r"[a-z]+", t.replace("{A}", "").replace("{B}", "")))
    words |= set(SHAPES) | set(COLORS) | {s for s in SIZES if s}
    return [PAD, UNK, *sorted(words)]


def tokenize(text: str, vocab: list[str]) -> tuple[list[int], list[str]]:
    """Lower-case word tokens -> ids (truncated to MAX_TOKENS). Returns (ids, unknown_words)."""
    idx = {w: i for i, w in enumerate(vocab)}
    words = re.findall(r"[a-z]+", text.lower())[:MAX_TOKENS]
    unknown = [w for w in words if w not in idx]
    return [idx.get(w, 1) for w in words] or [1], unknown


def batch_tokens(texts: list[str], vocab: list[str], device="cpu"):
    ids = [tokenize(t, vocab)[0] for t in texts]
    L = max(len(i) for i in ids)
    tok = torch.zeros(len(ids), L, dtype=torch.long)
    for b, i in enumerate(ids):
        tok[b, :len(i)] = torch.tensor(i)
    return tok.to(device), (tok != 0).to(device)


class FourierFeatures(nn.Module):
    def __init__(self, n_freq: int = 6):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        self.out_dim = 3 + 6 * n_freq

    def forward(self, x):
        xf = x[..., None] * self.freqs
        return torch.cat([x, torch.sin(xf).flatten(-2), torch.cos(xf).flatten(-2)], -1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d: int = 64, z_dim: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(MAX_TOKENS, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.proj = nn.Linear(d, z_dim)

    def forward(self, tok, mask, emb_delta: torch.Tensor | None = None):
        e = self.emb(tok)
        if emb_delta is not None:
            e = e + emb_delta[tok]
        h = self.tf(e + self.pos[: tok.shape[1]], src_key_padding_mask=~mask)
        m = mask[..., None].float()
        return self.proj((h * m).sum(1) / m.sum(1).clamp(min=1))


class SDFDecoder(nn.Module):
    def __init__(self, z_dim: int = 128, hidden: int = 256, n_freq: int = 6):
        super().__init__()
        self.ff = FourierFeatures(n_freq)
        din = self.ff.out_dim + z_dim
        self.l0 = nn.Linear(din, hidden)
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden + din, hidden)   # DeepSDF-style skip
        self.l3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 4)
        self.act = nn.SiLU()

    def forward(self, pts, z):
        zin = z[:, None, :].expand(-1, pts.shape[1], -1)
        x = torch.cat([self.ff(pts), zin], -1)
        h = self.act(self.l0(x))
        h = self.act(self.l1(h))
        h = self.act(self.l2(torch.cat([h, x], -1)))
        h = self.act(self.l3(h))
        o = self.out(h)
        return o[..., 0], o[..., 1:]                 # sdf, colour logits


class TextShapeModel(nn.Module):
    def __init__(self, vocab: list[str], d: int = 64, z_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.vocab = list(vocab)
        self.hparams = {"d": d, "z_dim": z_dim, "hidden": hidden}
        self.encoder = TextEncoder(len(vocab), d, z_dim)
        self.decoder = SDFDecoder(z_dim, hidden)
        self.emb_delta: nn.Parameter | None = None   # set by apply_adapter

    def encode(self, texts: list[str]):
        tok, mask = batch_tokens(texts, self.vocab, self.encoder.emb.weight.device)
        return self.encoder(tok, mask, self.emb_delta)

    def forward(self, texts: list[str], pts):
        return self.decoder(pts, self.encode(texts))


# ------------------------------------------------------------------------------------------------- LoRA
LORA_TARGETS = ("encoder.proj", "decoder.l0", "decoder.l1", "decoder.l2", "decoder.l3", "decoder.out",
                "encoder.tf.layers.0.linear1", "encoder.tf.layers.0.linear2",
                "encoder.tf.layers.1.linear1", "encoder.tf.layers.1.linear2")


class LoRALinear(nn.Module):
    """y = W x + b + (alpha/r) * B A x ; W, b frozen; A ~ Kaiming, B = 0 (adapter starts as identity)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * (self.alpha / self.r)

    # Some callers read .weight/.bias directly instead of calling forward — notably the fused inference fast path
    # of nn.TransformerEncoderLayer in eval mode. Exposing the MERGED parameters keeps both paths identical, so an
    # adapter can never be silently ignored at inference.
    @property
    def weight(self):
        return self.base.weight + (self.alpha / self.r) * (self.B @ self.A)

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def _get(model, path):
    m = model
    for p in path.split("."):
        m = getattr(m, p) if not p.isdigit() else m[int(p)]
    return m


def _set(model, path, new):
    parent, last = path.rsplit(".", 1)
    p = _get(model, parent)
    if last.isdigit():
        p[int(last)] = new
    else:
        setattr(p, last, new)


def apply_adapter(model: TextShapeModel, r: int, alpha: float, new_tokens: list[str], targets=LORA_TARGETS):
    """Freeze the base, insert LoRA modules and an embedding-delta table (non-zero only for ``new_tokens``)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for t in targets:
        base = _get(model, t)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{t} is not nn.Linear")
        _set(model, t, LoRALinear(base, r, alpha))
    model.emb_delta = nn.Parameter(torch.zeros_like(model.encoder.emb.weight))
    ids = [model.vocab.index(w) for w in new_tokens]
    mask = torch.zeros(len(model.vocab), 1)
    mask[ids] = 1.0
    model.register_buffer("emb_delta_mask", mask)
    model.emb_delta.register_hook(lambda g: g * model.emb_delta_mask)   # only new-token rows ever change
    model.adapter_config = {"type": "lora+token_embedding_delta", "r": r, "alpha": alpha, "targets": list(targets),
                            "new_tokens": list(new_tokens)}
    return [p for p in model.parameters() if p.requires_grad]


def adapter_state(model: TextShapeModel) -> dict:
    sd = {k: v.detach().clone() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}
    sd["emb_delta"] = model.emb_delta.detach().clone()
    return sd


def count_params(model) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": tot, "trainable": tr}
