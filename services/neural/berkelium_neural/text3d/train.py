"""Training for BerkeliumTextShape: base model (from scratch) or LoRA adapter (frozen base).

Reproducibility: fixed seeds per step, deterministic torch algorithms, pinned thread count, config hash, resumable
atomic checkpoints (model + optimizer + step + RNG), metrics.jsonl and a resource receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from ..contracts import canonical_json, sha256_hex
from .dataset import composition_of, load_manifest
from .model import TextShapeModel, adapter_state, apply_adapter, build_vocab, count_params
from .shapes import TEMPLATES, Composition, colors_at, sdf

TRAIN_VERSION = "berkelium-textshape-train/1"


def config_hash(cfg: dict) -> str:
    keep = {k: v for k, v in cfg.items() if k not in ("out_dir", "resume", "max_steps_this_run")}
    return sha256_hex(canonical_json(keep))


def sample_points(c: Composition, n: int, rng: np.random.Generator, near_frac=0.6, sigmas=(0.01, 0.05)):
    """Uniform points in [-1,1]^3 plus near-surface points (uniform points projected onto the surface with the
    exact SDF gradient, then jittered). Returns pts [n,3], sdf [n], colours [n,3]."""
    n_near = int(n * near_frac)
    u = rng.uniform(-1, 1, (n - n_near, 3))
    q = rng.uniform(-1, 1, (n_near * 2, 3))
    d = sdf(c, q)
    e = 1e-4
    g = np.stack([(sdf(c, q + e * np.eye(3)[k]) - sdf(c, q - e * np.eye(3)[k])) / (2 * e) for k in range(3)], 1)
    surf = q - d[:, None] * g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    surf = surf[np.abs(sdf(c, surf)) < 5e-3][:n_near]
    while len(surf) < n_near:                        # extremely unlikely; pad deterministically
        surf = np.concatenate([surf, surf[: n_near - len(surf)]]) if len(surf) else u[:n_near]
    sig = rng.choice(sigmas, size=(n_near, 1))
    near = surf + rng.normal(size=(n_near, 3)) * sig
    pts = np.concatenate([u, near]).astype(np.float32)
    return pts, sdf(c, pts).astype(np.float32), colors_at(c, pts).astype(np.float32)


def make_batch(items: list[tuple[Composition, str]], step: int, cfg: dict):
    rng = np.random.default_rng([cfg["seed"], step])
    sel = rng.choice(len(items), size=cfg["batch_assets"], replace=len(items) < cfg["batch_assets"])
    texts, P, D, C = [], [], [], []
    for i in sel:
        c, cap = items[i]
        if cfg.get("paraphrase_augment", True):     # same asset, another grammar template (no new asset)
            t = int(rng.integers(len(TEMPLATES[c.relation])))
            cap = Composition(c.relation, c.parts, t).caption()
        p, d, col = sample_points(c, cfg["points_per_asset"], rng)
        texts.append(cap)
        P.append(p)
        D.append(d)
        C.append(col)
    return texts, torch.from_numpy(np.stack(P)), torch.from_numpy(np.stack(D)), torch.from_numpy(np.stack(C))


def loss_fn(model, texts, pts, gt, gtc, delta=0.1, w_color=0.5):
    pred, logit = model(texts, pts)
    l_sdf = (pred.clamp(-delta, delta) - gt.clamp(-delta, delta)).abs().mean()
    near = (gt.abs() < 0.05).float()
    l_col = ((torch.sigmoid(logit) - gtc).abs().mean(-1) * near).sum() / near.sum().clamp(min=1)
    return l_sdf + w_color * l_col, {"sdf_l1": l_sdf.item(), "color_l1": l_col.item()}


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt: Path, adapter: Path | None = None, device="cpu") -> TextShapeModel:
    st = torch.load(ckpt, map_location=device, weights_only=False)
    m = TextShapeModel(st["vocab"], **st["hparams"]).to(device)
    m.load_state_dict(st["model"])
    m.identity = {"base_checkpoint_sha256": file_sha256(Path(ckpt)), "train_version": st.get("train_version")}
    if adapter is not None:
        ad = torch.load(adapter, map_location=device, weights_only=False)
        if ad["base_checkpoint_sha256"] != m.identity["base_checkpoint_sha256"]:
            raise ValueError("adapter was trained against a different base checkpoint")
        a = ad["adapter_config"]
        apply_adapter(m, a["r"], a["alpha"], a["new_tokens"], tuple(a["targets"]))
        missing = m.load_state_dict({**m.state_dict(), **ad["state"]}, strict=True)
        del missing
        m.identity["adapter_sha256"] = file_sha256(Path(adapter))
    m.eval()
    return m


def train(cfg: dict, log=print) -> dict:
    """cfg keys: kind (base|lora), dataset, train_splits, steps, batch_assets, points_per_asset, lr, seed,
    threads, out_dir, checkpoint_every, [base_checkpoint, lora_r, lora_alpha, new_tokens], model {d,z_dim,hidden}."""
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", 2))
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg["seed"])
    chash = config_hash(cfg)
    recs = load_manifest(Path(cfg["dataset"]), set(cfg["train_splits"]))
    items = [(composition_of(r), r.caption) for r in recs]
    if not items:
        raise ValueError(f"no training assets in splits {cfg['train_splits']}")
    if cfg["kind"] == "base":
        model = TextShapeModel(build_vocab(), **cfg.get("model", {}))
        params = list(model.parameters())
    elif cfg["kind"] == "lora":
        model = load_model(Path(cfg["base_checkpoint"]))
        model.train()
        params = apply_adapter(model, cfg["lora_r"], cfg["lora_alpha"], cfg.get("new_tokens", []))
    else:
        raise ValueError(cfg["kind"])
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, cfg["steps"]) / cfg["steps"])) * min(1.0, (s + 1) / 50))
    ck = out / "checkpoint.pt"
    step, elapsed_prev = 0, 0.0
    if ck.exists():                                   # resume
        st = torch.load(ck, map_location="cpu", weights_only=False)
        if st["config_hash"] != chash:
            raise ValueError("existing checkpoint was produced by a different config; refusing to resume")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["torch_rng"])
        step, elapsed_prev = st["step"], st["elapsed_s"]
        log(f"resumed at step {step}")
    pc = count_params(model)
    metrics = (out / "metrics.jsonl").open("a")
    t0, c0 = time.time(), time.process_time()
    stop_at = min(cfg["steps"], step + cfg.get("max_steps_this_run", cfg["steps"]))

    def save(path: Path, final: bool):
        state = {"model": model.state_dict(), "optim": opt.state_dict(), "sched": sched.state_dict(),
                 "torch_rng": torch.get_rng_state(), "step": step, "config": cfg, "config_hash": chash,
                 "elapsed_s": elapsed_prev + time.time() - t0, "vocab": model.vocab, "hparams": model.hparams,
                 "train_version": TRAIN_VERSION}
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    model.train()
    while step < stop_at:
        texts, pts, gt, gtc = make_batch(items, step, cfg)
        loss, parts = loss_fn(model, texts, pts, gt, gtc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 1.0))
        opt.step()
        sched.step()
        step += 1
        if step % cfg.get("log_every", 25) == 0 or step == stop_at:
            row = {"step": step, "loss": loss.item(), **parts, "grad_norm": float(gn), "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(elapsed_prev + time.time() - t0, 2)}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            log(json.dumps(row))
        if step % cfg.get("checkpoint_every", 200) == 0 or step == stop_at:
            save(ck, step == cfg["steps"])
    metrics.close()
    result = {"config_hash": chash, "step": step, "complete": step >= cfg["steps"], "params": pc}
    if step >= cfg["steps"]:
        if cfg["kind"] == "base":
            final = out / "model.pt"
            torch.save({"model": model.state_dict(), "vocab": model.vocab, "hparams": model.hparams,
                        "train_version": TRAIN_VERSION, "config": cfg, "config_hash": chash}, final)
        else:
            final = out / "adapter.pt"
            torch.save({"state": adapter_state(model), "adapter_config": model.adapter_config,
                        "base_checkpoint_sha256": model.identity["base_checkpoint_sha256"], "config": cfg,
                        "config_hash": chash, "train_version": TRAIN_VERSION}, final)
        result.update({"artifact": str(final), "artifact_sha256": file_sha256(final),
                       "artifact_bytes": final.stat().st_size})
    card = json.loads((Path(cfg["dataset"]) / "dataset_card.json").read_text())
    result["receipt"] = {
        "wall_s": round(elapsed_prev + time.time() - t0, 2), "cpu_s_this_run": round(time.process_time() - c0, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "device": "cpu", "threads": torch.get_num_threads(), "torch": torch.__version__,
        "python": platform.python_version(), "platform": platform.platform(), "gpu_hours": 0.0,
        "dataset_manifest_sha256": card["manifest_sha256"], "dataset_split_hashes": card["split_hashes"],
        "n_train_assets": len(items), "seed": cfg["seed"]}
    (out / "receipt.json").write_text(json.dumps(result, indent=2))
    return result
