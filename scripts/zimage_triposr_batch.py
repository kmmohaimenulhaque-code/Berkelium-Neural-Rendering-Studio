#!/usr/bin/env python3
"""E-T3 units for the zimage-triposr generator, staged so that no two large models are ever resident together.

  python scripts/zimage_triposr_batch.py plan                         # unit list (same keys/dirs as E-T3)
  python scripts/zimage_triposr_batch.py fetch                        # pinned, hash-verified weights
  python scripts/zimage_triposr_batch.py embed                        # Qwen3 text encoder -> all anchor prompts
  python scripts/zimage_triposr_batch.py anchors --shard 0/2          # Z-Image-Turbo -> anchor.png per unit
  .venv-triposr-cuda/bin/python scripts/zimage_triposr_batch.py reconstruct --shard 0/2   # key + TripoSR + clean

A unit is (generator, prompt, seed) exactly as in scripts/run_openworld_benchmark.py; its folder is
results/text3d/E-T3/units/<generator>/<slug>__<hash>__s<seed>/ with anchor.png, anchor.json, anchor_rgba.png,
triposr_input.png, mesh.npz, mesh.glb, preview.png and report.json. The ledger row (units_<gen>_cuda_<host>[_shard]
.jsonl) is written once per unit when reconstruction ends, whatever the outcome: a rejected anchor or an invalid mesh
is a result, never retried or replaced. ``--lora`` + ``--gen`` produce the anchor-LoRA variant under its own name.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from _common import ROOT, provenance

sys.path.insert(0, str(ROOT / "scripts"))
import run_openworld_benchmark as rob  # noqa: E402

GROUPS = ["B1", "B2", "B3_all", "B4_single", "B4_planner", "B1_seed1", "B1_seed2"]
EMB = ROOT / "data" / "zimage_embeds"


def units_for(gen: str, groups=GROUPS):
    cfg, prompts = rob.load_cfg()
    units, _ = rob.plan_units(cfg, prompts, groups, gen=gen)
    return cfg, units


def shard(units, s):
    i, n = s
    return [u for j, u in enumerate(units) if j % n == i]


def emb_path(text: str) -> Path:
    return EMB / (hashlib.sha256(text.encode()).hexdigest()[:24] + ".pt")


def cmd_plan(a):
    _, units = units_for(a.gen)
    print(json.dumps({"generator": a.gen, "units": len(units),
                      "by_group": {g: sum(u["group"] == g for u in units) for g in GROUPS}}, indent=2))


def cmd_fetch(a):
    from berkelium_neural.text3d import zimage_triposr as zt
    rec = {"zimage": zt.fetch_zimage(), "triposr": zt.fetch_triposr()}
    print(json.dumps({k: len(v) for k, v in rec.items()}))


def cmd_embed(a):
    import torch

    from berkelium_neural.text3d import zimage_triposr as zt
    _, units = units_for(a.gen)
    texts = sorted({zt.anchor_prompt(u["prompt"])["text"] for u in units if _anchor_prompt_ok(u["prompt"])})
    todo = [t for t in texts if not emb_path(t).exists()]
    EMB.mkdir(parents=True, exist_ok=True)
    if a.te_dtype == "from-m5":                  # the text-encoder configuration the M5 bring-up measured
        ch = json.loads((ROOT / "results/m5/E-A1_receipt.json").read_text())["text_encoder"]["chosen"]
        a.te_dtype, a.device = ch["dtype"], ch["device"]
    t0 = time.time()
    for t, e in zip(todo, zt.encode_prompts(todo, a.device, dtype=a.te_dtype), strict=True):
        torch.save({"text": t, "embeds": e, "dtype": a.te_dtype}, emb_path(t))
    print(f"embedded {len(todo)} of {len(texts)} anchor prompts in {time.time() - t0:.0f}s ({a.te_dtype}, {a.device})")


def _anchor_prompt_ok(prompt):
    from berkelium_neural.text3d import zimage_triposr as zt
    try:
        zt.anchor_prompt(prompt)
        return True
    except zt.Rejected:
        return False


def cmd_anchors(a):
    import torch
    from PIL import Image

    from berkelium_neural.text3d import zimage_triposr as zt
    cfg, units = units_for(a.gen)
    out = ROOT / cfg["out"]
    mine = shard(units, a.shard)
    if a.dtype == "from-m5":                     # the dtype the M5 precision study measured and chose
        a.dtype = json.loads((ROOT / "results/m5/E-A1_receipt.json").read_text())["precision_study"]["chosen"]
    gen = zt.AnchorGenerator(a.device, a.dtype)
    if a.lora:
        gen.pipe.load_lora_weights(a.lora)
        lora_rec = {"path": a.lora, "sha256": zt.sha256_file(Path(a.lora))}
    print(f"{len(mine)} anchors on {a.device} ({a.dtype}); load {gen.load_s}s", flush=True)
    for u in mine:
        d = rob.unit_dir(out, a.gen, u["prompt"], u["seed"])
        if (d / "anchor.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        try:
            ap = zt.anchor_prompt(u["prompt"])
        except zt.Rejected as e:
            (d / "anchor.json").write_text(json.dumps({"status": "rejected", "reason": str(e)}, indent=2))
            continue
        e = torch.load(emb_path(ap["text"]), weights_only=True)["embeds"]
        img, st = gen(e, u["seed"], zt.SAMPLER["steps"], zt.SAMPLER["height"])
        Image.fromarray((np.clip(np.nan_to_num(img), 0, 1) * 255).round().astype(np.uint8)).save(d / "anchor.png")
        rec = {"status": "ok" if st["finite"] and st["std"] > 0.02 else "blank_or_nonfinite", "anchor_prompt": ap,
               "sampler": zt.SAMPLER, "dtype": a.dtype, "device": a.device,
               "gpu": torch.cuda.get_device_name(a.device), "load_s": gen.load_s, **st,
               "anchor_sha256": zt.sha256_file(d / "anchor.png"), **({"lora": lora_rec} if a.lora else {})}
        (d / "anchor.json").write_text(json.dumps(rec, indent=2))
        print(f"[{u['group']}] {u['prompt']!r} s{u['seed']}: {rec['status']} {st['seconds']}s", flush=True)


def cmd_reconstruct(a):
    import torch
    from PIL import Image

    from berkelium_neural.camera import orbit_cameras
    from berkelium_neural.fitting.real_object import contact_sheet, save_png
    from berkelium_neural.text3d import zimage_triposr as zt
    from berkelium_neural.text3d.generate import mesh_to_glb
    from berkelium_neural.text3d.geometry import render_mesh
    cfg, units = units_for(a.gen)
    out = ROOT / cfg["out"]
    mine = shard(units, a.shard)
    ledger = f"units_{a.gen}_cuda_{platform.node()}" + (f"_shard{a.shard[0]}of{a.shard[1]}" if a.shard[1] > 1
                                                       else "") + ".jsonl"
    done = rob.read_units(out)
    recon = zt.Reconstructor(a.device)
    hw = {"device": "cuda", "gpu_name": torch.cuda.get_device_name(a.device), "host": platform.node(),
          "torch": torch.__version__, "torch_cuda": torch.version.cuda,
          "transformers": __import__("transformers").__version__}
    print(f"{len(mine)} units; TripoSR load {recon.load_s}s -> {ledger}", flush=True)
    for u in mine:
        key = rob.unit_key(a.gen, u["prompt"], u["seed"])
        if key in done:
            continue
        d = rob.unit_dir(out, a.gen, u["prompt"], u["seed"])
        t0 = time.time()
        anc = json.loads((d / "anchor.json").read_text()) if (d / "anchor.json").exists() else None
        rep = {"prompt": u["prompt"], "seed": u["seed"], "generator": a.gen, "anchor": anc,
               "identity": zt.ZImageTripoSRGenerator(dtype=(anc or {}).get("dtype", "float16")).identity,
               "artifact_kind": "triangle_mesh", "units": "normalized_object",
               "frame": "TripoSR z-up -> +Y-up (to_gradio_3d_orientation), centred, max extent 1.6",
               "labels": "model-generated, normalized units, not measured geometry", "hardware": hw}
        status = None
        if anc is None:
            status, rep["reason"] = "anchor_missing", "no anchor.json (anchor stage did not reach this unit)"
        elif anc["status"] != "ok":
            status, rep["reason"] = f"anchor_{anc['status']}", anc.get("reason", anc["status"])
        else:
            rgb = np.asarray(Image.open(d / "anchor.png").convert("RGB"), np.float32) / 255.0
            try:
                o = zt.mesh_from_anchor(rgb, anc["anchor_prompt"]["background_rgb"], recon)
            except zt.Rejected as e:
                status, rep["reason"] = "anchor_rejected", str(e)
            else:
                Image.fromarray((o["rgba"] * 255).round().astype(np.uint8), "RGBA").save(d / "anchor_rgba.png")
                o["tsr_input"].save(d / "triposr_input.png")
                rep.update(o["report"])
                if o["vertices"] is not None and rep.get("valid"):
                    np.savez_compressed(d / "mesh.npz", v=o["vertices"], f=o["faces"], col=o["colors"])
                    (d / "mesh.glb").write_bytes(mesh_to_glb(o["vertices"], o["faces"], o["colors"]))
                    save_png(d / "preview.png", contact_sheet([[render_mesh(o["vertices"], o["faces"], o["colors"], c)[0]
                                                                for c in orbit_cameras(4, 3.2, 25.0, 192, 192)]]))
                    status = "ok"
                else:
                    status = "invalid_mesh"
        recon_s = round(time.time() - t0, 2)
        rep.update({"status": status, "reconstruct_s": recon_s,
                    "seconds": round(((anc or {}).get("seconds") or 0) + recon_s, 2), "environment": provenance()})
        (d / "report.json").write_text(json.dumps(rep, indent=2, default=str))
        row = {"key": key, **u, "status": status, "dir": str(d.relative_to(ROOT)), "valid": status == "ok",
               "seconds": rep["seconds"], "anchor_s": (anc or {}).get("seconds"), "reconstruct_s": recon_s,
               "peak_gpu_mib": (anc or {}).get("peak_gpu_mib"), "device": "cuda", "hardware": hw,
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        rob.append_unit(out, row, ledger)
        print(f"[{u['group']}] {u['prompt']!r} s{u['seed']}: {status} ({recon_s}s)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["plan", "fetch", "embed", "anchors", "reconstruct"])
    ap.add_argument("--gen", default="zimage-triposr", help="generator name recorded in the ledger")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16", help="anchors: transformer dtype, or from-m5 (the M5 precision study's choice)")
    ap.add_argument("--te-dtype", default="bfloat16", help="embed: text encoder dtype, or from-m5 (also takes the M5 device)")
    ap.add_argument("--lora", default=None, help="anchors: anchor-LoRA weights (safetensors) for a variant")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(argv)
    a.shard = tuple(int(x) for x in a.shard.split("/"))
    {"plan": cmd_plan, "fetch": cmd_fetch, "embed": cmd_embed, "anchors": cmd_anchors,
     "reconstruct": cmd_reconstruct}[a.phase](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
