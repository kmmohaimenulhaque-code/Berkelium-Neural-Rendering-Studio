#!/usr/bin/env python3
"""Canonical 3D-anchor LoRA for Z-Image-Turbo (Phase II.2): teach the anchor stage to draw what TripoSR reconstructs
best, a single complete object in a canonical three-quarter view on a plain chroma background with even lighting.

  python scripts/anchor_lora.py pin        # this host: download the GSO zips once, record their sha256 in the config
  python scripts/anchor_lora.py data       # download + verify + render the training images (CPU)
  python scripts/anchor_lora.py prepare    # GPU: VAE latents + Qwen3 embeddings of the anchor captions, then freed
  python scripts/anchor_lora.py train      # GPU: LoRA on the frozen fp16 transformer -> exports/zimage-anchor-lora-v1
  python scripts/anchor_lora.py sample     # GPU: base vs LoRA anchors on held-out prompts (visual record only)

Data (configs/datasets/anchor_lora_gso_v1.json): 36 Google Scanned Objects (CC-BY 4.0) whose categories appear
nowhere in E-T3 B1/B2, 4 three-quarter views each at 512², albedo x (ambient + Lambert) shading, composited on the
chroma background farthest from the scan's own colours; the caption is the anchor template (as sent at inference)
naming that background. Training objective: the rectified-flow target of Z-Image's own sampler
(x_t = (1-s) x0 + s e; the transformer is given 1-s and predicts x0 - e, which pipeline_z_image negates for its Euler
step), s drawn from the 9-step Turbo schedule at the training resolution, MSE, fp16 autocast with fp32 LoRA weights
and a grad scaler.
Training does not change E-T3: the adapter enters the router only if its own E-T3 units beat the base generator
(decision rule written in NEURAL_RENDERING_DECISIONS.md before those units run).
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from _common import ROOT, provenance

CFG = ROOT / "configs" / "datasets" / "anchor_lora_gso_v1.json"
DATA = ROOT / "data" / "anchor_lora_gso_v1"
OUT = ROOT / "exports" / "zimage-anchor-lora-v1"
VAL_PROMPTS = ["a red fire hydrant", "a wooden rocking horse", "a toy sailboat", "an old typewriter"]  # not in E-T3


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fetch(name: str) -> Path:
    import urllib.request

    from berkelium_neural.fitting.real_object import GSO_BASE
    z = ROOT / "data" / "gso" / name / f"{name}.zip"
    if not z.exists():
        z.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{GSO_BASE}/{name}/1/{name}.zip", z.with_suffix(".part"))
        z.with_suffix(".part").rename(z)
    return z


def cmd_pin(a):
    cfg = json.loads(CFG.read_text())
    for o in cfg["objects"]:
        h = sha256(fetch(o["name"]))
        if o["zip_sha256"] and o["zip_sha256"] != h:
            raise SystemExit(f"{o['name']}: download {h} differs from the pinned {o['zip_sha256']}")
        o["zip_sha256"] = h
    CFG.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"pinned {len(cfg['objects'])} objects in {CFG.relative_to(ROOT)}")


def cmd_data(a):
    from PIL import Image

    from berkelium_neural.camera import make_camera
    from berkelium_neural.fitting.real_object import load_scan, render_scan
    from berkelium_neural.text3d import zimage_triposr as zt
    from berkelium_neural.text3d.geometry import render_mesh
    cfg = json.loads(CFG.read_text())
    vw = cfg["views"]
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for o in cfg["objects"]:
        z = fetch(o["name"])
        if sha256(z) != o["zip_sha256"]:
            raise SystemExit(f"{o['name']}: zip sha256 does not match the pin; refusing")
        scan = load_scan(z, tex_size=1024)
        views = []
        for az in vw["azimuths_deg"]:
            el, r = math.radians(vw["elevation_deg"]), vw["radius"]
            eye = (r * math.cos(el) * math.sin(math.radians(az)), r * math.sin(el), r * math.cos(el) *
                   math.cos(math.radians(az)))
            cam = make_camera(eye, (0, 0, 0), vw["size"], vw["size"], fov_y_deg=vw["fov_y_deg"])
            alb, alpha = render_scan(scan, cam, ss=2)[:2]
            shade, _, _ = render_mesh(scan.v, scan.f, np.ones((len(scan.v), 3), np.float32), cam,
                                      background=(1.0, 1.0, 1.0))
            fg = np.where(alpha[..., None] > 1e-4, alb / np.maximum(alpha[..., None], 1e-4), 0) * shade
            views.append((az, fg, alpha))
        # Background: the candidate that the fewest object pixels are close to (the scan's real colours, not the
        # caption, decide: a green toy is never drawn on green). The caption then names that background.
        px = np.concatenate([fg[a > 0.5] for _, fg, a in views])
        close = {n: float((np.linalg.norm(px - np.array(rgb) / 255.0, axis=1) < zt.MIN_BG_DISTANCE).mean())
                 for n, rgb in zt.BACKGROUNDS}
        name, rgb = min(zt.BACKGROUNDS, key=lambda b: (close[b[0]], zt.BACKGROUNDS.index(b)))
        text = zt.TEMPLATE.format(prompt=o["caption"].strip().rstrip("."), bg=name)
        bg = np.array(rgb, np.float32) / 255.0
        for az, fg, alpha in views:
            img = fg * alpha[..., None] + bg * (1 - alpha[..., None])
            p = DATA / f"{o['name']}__az{az:03d}.png"
            Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(p)
            rows.append({"image": p.name, "object": o["name"], "caption": o["caption"], "anchor_prompt": text,
                         "background": name, "object_pixels_near_background": close, "azimuth_deg": az,
                         "elevation_deg": vw["elevation_deg"], "zip_sha256": o["zip_sha256"], "png_sha256": sha256(p)})
        print(f"{o['name']}: {len(vw['azimuths_deg'])} views", flush=True)
    man = {"dataset": cfg["name"], "config_sha256": sha256(CFG), "license": cfg["source"], "n_images": len(rows),
           "rows": rows}
    (DATA / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"{len(rows)} images, manifest sha256 {sha256(DATA / 'manifest.json')[:16]}")


def cmd_prepare(a):
    import gc

    import torch
    from diffusers import AutoencoderKL
    from PIL import Image

    from berkelium_neural.text3d import zimage_triposr as zt
    man = json.loads((DATA / "manifest.json").read_text())
    texts = sorted({r["anchor_prompt"] for r in man["rows"]})
    emb = dict(zip(texts, zt.encode_prompts(texts, a.device, dtype=a.te_dtype), strict=True))
    vae = AutoencoderKL.from_pretrained(zt.zimage_dir() / "vae", torch_dtype=torch.float32).to(a.device).eval()
    sf, sh = vae.config.scaling_factor, vae.config.shift_factor or 0.0
    items = []
    with torch.no_grad():
        for r in man["rows"]:
            if sha256(DATA / r["image"]) != r["png_sha256"]:
                raise SystemExit(f"{r['image']} changed since the manifest was written")
            x = torch.from_numpy(np.asarray(Image.open(DATA / r["image"]).convert("RGB"), np.float32) / 127.5 - 1)
            lat = vae.encode(x.permute(2, 0, 1)[None].to(a.device)).latent_dist.mode()
            items.append({"latent": ((lat - sh) * sf)[0].float().cpu(), "emb": emb[r["anchor_prompt"]],
                          "image": r["image"]})
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    torch.save({"items": items, "manifest_sha256": sha256(DATA / "manifest.json"), "te_dtype": a.te_dtype},
               DATA / "prepared.pt")
    print(f"prepared {len(items)} latents {tuple(items[0]['latent'].shape)}, {len(texts)} captions")


def turbo_sigmas(seq_len: int, steps: int = 9):
    """The sigmas the Turbo sampler visits at this resolution (pipeline_z_image: linspace(1, 1/steps) + shift mu)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, get_default_z_image_sigmas

    from berkelium_neural.text3d import zimage_triposr as zt
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(zt.zimage_dir() / "scheduler")
    mu = calculate_shift(seq_len, sch.config.get("base_image_seq_len", 256), sch.config.get("max_image_seq_len", 4096),
                         sch.config.get("base_shift", 0.5), sch.config.get("max_shift", 1.15))
    sch.set_timesteps(sigmas=get_default_z_image_sigmas(steps), mu=mu)
    return [float(s) for s in sch.sigmas[:-1]], mu


def build_lora(transformer, rank: int):
    from peft import LoraConfig
    transformer.requires_grad_(False)
    transformer.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                       target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    params = []
    for _n, p in transformer.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()                   # fp32 master weights for the adapter only
            params.append(p)
    return params


def train_loop(tr, items, sigmas, steps, lr, seed, device, log, use_amp=True, ckpt_every=0, save_fn=None):
    import torch
    import torch.nn.functional as F
    params = [p for p in tr.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    g = torch.Generator().manual_seed(seed)
    tr.train()
    for step in range(1, steps + 1):
        it = items[int(torch.randint(len(items), (1,), generator=g))]
        s = sigmas[int(torch.randint(len(sigmas), (1,), generator=g))]
        x0 = it["latent"].to(device)
        eps = torch.randn(x0.shape, generator=g).to(device)
        xt = (1 - s) * x0 + s * eps
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = tr([xt.unsqueeze(1).to(next(tr.parameters()).dtype if not use_amp else torch.float16)],
                     torch.tensor([1.0 - s], device=device), [it["emb"].to(device)], return_dict=False)[0][0]
        loss = F.mse_loss(out.float().squeeze(1), (x0 - eps).float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        log({"step": step, "loss": float(loss.detach()), "sigma": s, "grad_norm": float(gn.detach()),
             "scale": scaler.get_scale()})
        if ckpt_every and save_fn and step % ckpt_every == 0:
            save_fn(step)
    return params


def cmd_train(a):
    import torch
    from diffusers import ZImagePipeline, ZImageTransformer2DModel
    from peft.utils import get_peft_model_state_dict

    from berkelium_neural.text3d import zimage_triposr as zt
    prep = torch.load(DATA / "prepared.pt", weights_only=False)
    items = prep["items"]
    h, w = items[0]["latent"].shape[-2:]
    sigmas, mu = turbo_sigmas((h // 2) * (w // 2))
    t0 = time.time()
    tr = ZImageTransformer2DModel.from_pretrained(zt.zimage_dir() / "transformer", torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    params = build_lora(tr, a.rank)
    tr.enable_gradient_checkpointing()
    tr.to(a.device)
    load_s = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "loss.jsonl", "w")
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats(a.device)

    def log(row):
        row["elapsed_s"] = round(time.time() - t1, 1)
        logf.write(json.dumps(row) + "\n")
        if row["step"] % 25 == 0 or row["step"] == 1:
            logf.flush()
            print(json.dumps(row), flush=True)
    train_loop(tr, items, sigmas, a.steps, a.lr, a.seed, a.device, log)
    logf.close()
    sd = get_peft_model_state_dict(tr)
    ZImagePipeline.save_lora_weights(OUT, transformer_lora_layers={k: v.half() for k, v in sd.items()},
                                     weight_name="adapter.safetensors")
    losses = [json.loads(x)["loss"] for x in (OUT / "loss.jsonl").read_text().splitlines()]
    n = max(1, len(losses) // 10)
    card = {"name": "zimage-anchor-lora-v1", "base": zt.weight_identity()["zimage"],
            "kind": "LoRA (peft) on Z-Image-Turbo transformer attention (to_q, to_k, to_v, to_out.0), base frozen",
            "rank": a.rank, "alpha": a.rank, "trainable_params": int(sum(p.numel() for p in params)),
            "data": {"manifest_sha256": prep["manifest_sha256"], "images": len(items),
                     "config": str(CFG.relative_to(ROOT)),
                     "license": "CC-BY 4.0 (Google Scanned Objects); derived weights trained on renders"},
            "objective": "rectified-flow x0 - eps target at the 9 Turbo sigmas (training resolution shift mu "
                         f"{mu:.4f}), MSE", "steps": a.steps, "lr": a.lr, "seed": a.seed, "batch": 1,
            "precision": "fp16 frozen base + fp32 LoRA, autocast fp16, grad scaler, grad clip 1.0",
            "loss_first_10pct_mean": float(np.mean(losses[:n])), "loss_last_10pct_mean": float(np.mean(losses[-n:])),
            "weights": {"adapter.safetensors": sha256(OUT / "adapter.safetensors")},
            "license": "Apache-2.0 (base model) adapter weights released with the repository",
            "status": "trained; NOT promoted unless it wins E-T3 (NEURAL_RENDERING_DECISIONS.md D-030)"}
    (OUT / "MODEL_CARD.json").write_text(json.dumps(card, indent=2))
    receipt = {"load_s": round(load_s, 1), "train_s": round(time.time() - t1, 1),
               "peak_gpu_mib": round(torch.cuda.max_memory_allocated(a.device) / 2**20, 1),
               "gpu": torch.cuda.get_device_name(a.device), "environment": provenance(),
               "adapter_sha256": card["weights"]["adapter.safetensors"]}
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({**receipt, "loss_first": card["loss_first_10pct_mean"],
                      "loss_last": card["loss_last_10pct_mean"]}))


def cmd_sample(a):
    from PIL import Image

    from berkelium_neural.text3d import zimage_triposr as zt
    texts = [zt.anchor_prompt(p)["text"] for p in VAL_PROMPTS]
    emb = zt.encode_prompts(texts, a.device, dtype=a.te_dtype)
    gen = zt.AnchorGenerator(a.device, a.dtype)
    rows = []
    for tag in ("base", "lora"):
        if tag == "lora":
            gen.pipe.load_lora_weights(str(OUT / "adapter.safetensors"))
        for i, (p, e) in enumerate(zip(VAL_PROMPTS, emb, strict=True)):
            img, st = gen(e, 100 + i)
            f = OUT / "samples" / f"{tag}_{i}.png"
            f.parent.mkdir(exist_ok=True)
            Image.fromarray((np.clip(np.nan_to_num(img), 0, 1) * 255).round().astype(np.uint8)).save(f)
            try:
                _, ks = zt.key_background(np.asarray(Image.open(f).convert("RGB"), np.float32) / 255,
                                          zt.anchor_prompt(p)["background_rgb"])
                keyed = {"ok": True, **{k: ks[k] for k in ("foreground_fraction", "border_spread_p95")}}
            except zt.Rejected as err:
                keyed = {"ok": False, "reason": str(err)}
            rows.append({"variant": tag, "prompt": p, "file": str(f.relative_to(ROOT)), "keying": keyed, **st})
            print(tag, p, keyed, flush=True)
    (OUT / "samples" / "samples.json").write_text(json.dumps(rows, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["pin", "data", "prepare", "train", "sample"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--te-dtype", default="bfloat16")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.te_dtype == "from-m5" or a.dtype == "from-m5":
        m5 = json.loads((ROOT / "results/m5/E-A1_receipt.json").read_text())
        if a.te_dtype == "from-m5":
            a.te_dtype = m5["text_encoder"]["chosen"]["dtype"]
        if a.dtype == "from-m5":
            a.dtype = m5["precision_study"]["chosen"]
    {"pin": cmd_pin, "data": cmd_data, "prepare": cmd_prepare, "train": cmd_train, "sample": cmd_sample}[a.phase](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
