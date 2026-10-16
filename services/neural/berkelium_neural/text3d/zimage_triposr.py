"""zimage-triposr: text -> canonical anchor image (Z-Image-Turbo) -> keyed RGBA -> TripoSR mesh (decision D-006).

Four stages, each a separate, inspectable artifact, never two large models resident at once:

  1. anchor prompt   The user's object prompt inside a fixed canonical-view template (single complete object,
                     three-quarter view, plain solid chroma background chosen to contrast with the colours the prompt
                     names). Deterministic; no LLM.
  2. anchor image    Z-Image-Turbo @ f332072a (Apache-2.0), 9 scheduler steps, CFG 0 (upstream Turbo settings); the
                     initial latent noise is drawn from a seeded CPU generator, so a seed means the same draw on any
                     device. The text encoder runs first and is freed before the transformer loads.
  3. keying          Background removal by colour distance to the measured border colour (no segmentation network),
                     largest component + attached parts, edge de-spill. Gated: an anchor whose object touches the frame,
                     whose background is not uniform, or whose foreground is implausibly small/large is REJECTED with
                     the reason (the unit fails; nothing is retried or substituted).
  4. reconstruction  TripoSR @ 107cefd code / 5b521936 weights (MIT), the M6 path (D-020: scikit-image marching
                     cubes behind the torchmcubes shim), TripoSR's z-up -> +Y-up, centred, max extent 1.6, then the
                     float32-first clean (round to the returned float32 coordinates, then drop zero-area faces), so the
                     validator's degenerate-face count is 0 by construction.

Every weight file is verified against its pinned hash when fetched (sha256 for LFS files, git blob sha1 for small
files: both are the hashes the Hugging Face API publishes for the pinned revision); the per-file record goes into the
generator identity and every report. Derived files (the fp16 transformer) are not written: weights are cast at load.

The output is model-generated geometry in normalized units, not measured or CAD geometry. The anchor image is a
generated picture with no calibrated camera; TripoSR's pose prior (level camera) is an assumption, not a measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
NAME = "zimage-triposr"
EXTENT = 1.6
DEFAULT_CACHE = Path(os.environ.get("BERKELIUM_MODEL_CACHE", REPO / "data" / "models"))

ZIMAGE = {
    "repo": "Tongyi-MAI/Z-Image-Turbo", "revision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
    "license": "Apache-2.0 (model card metadata at the pinned revision)",
    # path -> (hash kind, hash, bytes), from https://huggingface.co/api/models/Tongyi-MAI/Z-Image-Turbo/tree/<rev>
    "files": {
        "model_index.json": ("git_sha1", "570c63a7c5580f63d0a8ed622324f50015dcf914", 467),
        "scheduler/scheduler_config.json": ("git_sha1", "2349bc43351afa730341d07dd44fce4ace1c4257", 173),
        "text_encoder/config.json": ("git_sha1", "e49eccdc32f36da9c09cfa0e737084f9e0105e5e", 726),
        "text_encoder/generation_config.json": ("git_sha1", "20a8a9156fc8c3f25295ca067f61fdf120d517c5", 239),
        "text_encoder/model-00001-of-00003.safetensors":
            ("sha256", "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223", 3957900840),
        "text_encoder/model-00002-of-00003.safetensors":
            ("sha256", "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5", 3987450520),
        "text_encoder/model-00003-of-00003.safetensors":
            ("sha256", "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52", 99630640),
        "text_encoder/model.safetensors.index.json": ("git_sha1", "95c0a0059df040d75dc6c396b174382cf61d2f91", 32819),
        "tokenizer/merges.txt": ("git_sha1", "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7", 1671853),
        "tokenizer/tokenizer.json":
            ("sha256", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", 11422654),
        "tokenizer/tokenizer_config.json": ("git_sha1", "417d038a63fa3de29cfde265caedae14d1a58d92", 9732),
        "tokenizer/vocab.json": ("git_sha1", "4783fe10ac3adce15ac8f358ef5462739852c569", 2776833),
        "transformer/config.json": ("git_sha1", "c579190a5e03d602a2fd9647221c9d0d9441f150", 473),
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors":
            ("sha256", "95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804", 9973693184),
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors":
            ("sha256", "a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015", 9973714824),
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors":
            ("sha256", "aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa", 4672282880),
        "transformer/diffusion_pytorch_model.safetensors.index.json":
            ("git_sha1", "a3cadc1ed28e334ef58d4e6bbda107e25168ef47", 48969),
        "vae/config.json": ("git_sha1", "894fa30ac1950cef422189d2d4cef11043c62875", 805),
        "vae/diffusion_pytorch_model.safetensors":
            ("sha256", "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3", 167666902),
    },
}
TRIPOSR = {
    "code_repo": "https://github.com/VAST-AI-Research/TripoSR.git",
    "code_commit": "107cefdc244c39106fa830359024f6a2f1c78871",
    "weights_repo": "stabilityai/TripoSR", "revision": "5b521936b01fbe1890f6f9baed0254ab6351c04a",
    "license": "MIT (upstream statement for code and weights)",
    "files": {"model.ckpt": ("sha256", "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee", 1677246742),
              "config.yaml": ("git_sha1", "fa3e59a701f02679bd1dca02fbaf97306174d291", 987)},
}
# 768² (2304 image tokens), not the upstream 1024²: on a 15 GB T4 with the fp16 transformer resident (12.3 GB) the
# attention fell back to the math kernel and one 30-head 4096² score matrix (1-1.9 GiB) did not fit (M5 attempt 1,
# results/m5/E-A1_receipt.attempt1_1024_oom.json). TripoSR consumes 512² anyway.
SAMPLER = {"steps": 9, "guidance_scale": 0.0, "height": 768, "width": 768, "noise": "seeded CPU generator"}
TRIPOSR_CFG = {"input_size": 512, "foreground_ratio": 0.85, "mc_resolution": 256, "chunk_size": 8192}
# Stage 3, v2 (2026-10-16, after the v1 E-T3 anchors): U²-Net salient-object segmentation, the network TripoSR's own
# preprocessing uses (through rembg). v1's colour keying assumed a flat background; Z-Image draws studio backdrops
# with gradients and vignetting, so 33 of 80 v1 anchors were rejected as "background not uniform" (E-T3 ledger
# units_zimage-triposr_*). The change was made after seeing those rejections and before any geometry was scored.
U2NET = {"source": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
         "sha256": "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491", "bytes": 175997641,
         "license": "Apache-2.0 (U^2-Net, Qin et al. 2020; ONNX export distributed by rembg, MIT)"}

# Anchor background: the candidate farthest (RGB distance) from every colour the prompt names, in preference order on
# ties; refused if even the best is closer than MIN_BG_DISTANCE to a named colour (keying would eat the object).
BACKGROUNDS = [("bright green", (0, 177, 64)), ("bright magenta", (255, 0, 255)), ("bright blue", (0, 71, 187))]
COLOR_WORDS = {"red": (200, 30, 30), "orange": (240, 130, 20), "yellow": (240, 220, 30), "green": (40, 160, 50),
               "cyan": (0, 200, 220), "blue": (30, 70, 200), "purple": (130, 40, 170), "white": (240, 240, 240),
               "black": (20, 20, 20), "grey": (128, 128, 128), "gray": (128, 128, 128), "brown": (120, 70, 30),
               "pink": (240, 130, 180), "teal": (0, 128, 128), "beige": (220, 200, 160), "magenta": (230, 0, 230),
               "violet": (140, 60, 200), "navy": (20, 30, 110), "lime": (150, 230, 40), "gold": (212, 175, 55),
               "silver": (190, 190, 200), "olive": (110, 120, 30), "turquoise": (60, 210, 200)}
MIN_BG_DISTANCE = 0.35       # in [0, sqrt(3)], RGB scaled to [0, 1]
TEMPLATE = ("{prompt}. A single complete object, centered, fully visible with empty margin around it, "
            "three-quarter view from slightly above, isolated on a plain solid {bg} background, "
            "soft even studio lighting, no shadow, no text, no other objects")


class Rejected(RuntimeError):
    """A stage refused its input with a recorded reason (the unit fails; nothing is substituted)."""


# ------------------------------------------------------------------------------------------------- hashing / fetch
def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    data = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_file(p: Path, kind: str, expected: str, size: int) -> dict:
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.stat().st_size != size:
        raise ValueError(f"{p}: {p.stat().st_size} bytes, pinned {size}")
    got = sha256_file(p) if kind == "sha256" else git_blob_sha1(p)
    if got != expected:
        raise ValueError(f"{p}: {kind} {got} != pinned {expected}")
    return {"bytes": size, kind: got}


def _fetch(repo: str, revision: str, files: dict, dest: Path, only=None) -> dict:
    """Download each pinned file of ``repo@revision`` into ``dest`` (resumable), verify its hash, and write
    ``dest/verified.json`` (path -> bytes + hash). Raises on any mismatch; never keeps an unverified file."""
    from huggingface_hub import hf_hub_download
    dest.mkdir(parents=True, exist_ok=True)
    rec_path = dest / "verified.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    for path, (kind, h, size) in files.items():
        if only and not any(path.startswith(o) for o in only):
            continue
        p = dest / path
        if rec.get(path, {}).get(kind) == h and p.is_file() and p.stat().st_size == size:
            continue
        t0 = time.time()
        hf_hub_download(repo, path, revision=revision, local_dir=str(dest))
        try:
            rec[path] = {**verify_file(p, kind, h, size), "download_s": round(time.time() - t0, 1)}
        except ValueError:
            p.unlink(missing_ok=True)
            raise
        rec_path.write_text(json.dumps(rec, indent=2))
        print(f"verified {path} ({size / 2**30:.2f} GiB, {time.time() - t0:.0f}s)", flush=True)
    return rec


def zimage_dir(cache: Path | None = None) -> Path:
    return Path(cache or DEFAULT_CACHE) / "zimage_turbo" / ZIMAGE["revision"][:8]


def triposr_dir(cache: Path | None = None) -> Path:
    return Path(cache or DEFAULT_CACHE) / "triposr" / TRIPOSR["revision"][:8]


def fetch_zimage(cache: Path | None = None, only=None) -> dict:
    return _fetch(ZIMAGE["repo"], ZIMAGE["revision"], ZIMAGE["files"], zimage_dir(cache), only)


def fetch_triposr(cache: Path | None = None) -> dict:
    d = REPO / "third_party" / "TripoSR"
    if not (d / ".git").exists():
        subprocess.run(["git", "clone", "-q", TRIPOSR["code_repo"], str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", TRIPOSR["code_commit"]], check=True)
    return _fetch(TRIPOSR["weights_repo"], TRIPOSR["revision"], TRIPOSR["files"], triposr_dir(cache))


def _recorded(dest: Path, files: dict) -> tuple[bool, str]:
    """Cheap presence check for available(): every pinned file listed as verified and still the pinned size."""
    rp = dest / "verified.json"
    if not rp.exists():
        return False, f"weights not fetched/verified ({dest}); run scripts/zimage_triposr_batch.py fetch"
    rec = json.loads(rp.read_text())
    for path, (kind, h, size) in files.items():
        p = dest / path
        if rec.get(path, {}).get(kind) != h or not p.is_file() or p.stat().st_size != size:
            return False, f"{path} missing or not verified in {dest}"
    return True, ""


def weight_identity(cache: Path | None = None) -> dict:
    return {"zimage": {"repo": ZIMAGE["repo"], "revision": ZIMAGE["revision"], "license": ZIMAGE["license"],
                       "files": {p: {k: v for k, v in zip(("hash_kind", "hash", "bytes"), x, strict=True)}
                                 for p, x in ZIMAGE["files"].items() if p.endswith(".safetensors")}},
            "triposr": {"code_commit": TRIPOSR["code_commit"], "weights_revision": TRIPOSR["revision"],
                        "model_ckpt_sha256": TRIPOSR["files"]["model.ckpt"][1], "license": TRIPOSR["license"]}}


# --------------------------------------------------------------------------------------------- stage 1: prompt
def anchor_prompt(prompt: str) -> dict:
    import re
    named = [w for w in re.findall(r"[a-z]+", prompt.lower()) if w in COLOR_WORDS]

    def margin(rgb):
        return min((float(np.linalg.norm((np.array(rgb) - np.array(COLOR_WORDS[w])) / 255.0)) for w in named),
                   default=float("inf"))
    name, rgb = max(BACKGROUNDS, key=lambda b: (margin(b[1]), -BACKGROUNDS.index(b)))
    if margin(rgb) < MIN_BG_DISTANCE:
        raise Rejected(f"no background colour is at least {MIN_BG_DISTANCE} from every named colour {named}")
    return {"text": TEMPLATE.format(prompt=prompt.strip().rstrip("."), bg=name), "background": name,
            "background_rgb": list(rgb), "named_colours": named, "colour_margin": round(margin(rgb), 3),
            "template": TEMPLATE}


# ------------------------------------------------------------------------------------------- stage 2: Z-Image
def _dtype(name: str):
    import torch
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def encode_prompts(texts: list[str], device: str, cache: Path | None = None, dtype: str = "bfloat16") -> list:
    """Qwen3 text encoder (the class model_index.json names) -> per-prompt embeddings through the pipeline's own
    encode_prompt, returned on the CPU in float32. The encoder is loaded, used and released here, before the
    transformer is ever loaded."""
    import gc

    import torch
    from diffusers import ZImagePipeline
    from transformers import AutoTokenizer, Qwen3Model
    d = zimage_dir(cache)
    tok = AutoTokenizer.from_pretrained(d / "tokenizer")
    te = Qwen3Model.from_pretrained(d / "text_encoder", torch_dtype=_dtype(dtype), low_cpu_mem_usage=True).to(device)
    enc = ZImagePipeline(scheduler=None, vae=None, text_encoder=te, tokenizer=tok, transformer=None)
    out = []
    with torch.no_grad():
        for t in texts:
            pe, _ = enc.encode_prompt(t, device=device, do_classifier_free_guidance=False)
            out.append(pe[0].float().cpu())
    del enc, te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


class AnchorGenerator:
    """Z-Image-Turbo transformer + VAE on one device (text encoder never resident here)."""

    def __init__(self, device: str = "cuda", dtype: str = "float16", cache: Path | None = None):
        import torch
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
        d = zimage_dir(cache)
        t0 = time.time()
        tr = ZImageTransformer2DModel.from_pretrained(d / "transformer", torch_dtype=_dtype(dtype),
                                                      low_cpu_mem_usage=True).to(device)
        vae = AutoencoderKL.from_pretrained(d / "vae", torch_dtype=torch.float32).to(device)
        vae.enable_tiling()                       # bounded decode memory next to the resident transformer
        sch = FlowMatchEulerDiscreteScheduler.from_pretrained(d / "scheduler")
        self.pipe = ZImagePipeline(scheduler=sch, vae=vae, text_encoder=None, tokenizer=None, transformer=tr)
        self.pipe.set_progress_bar_config(disable=True)
        self.device, self.dtype = device, dtype
        torch.cuda.synchronize()
        self.load_s = round(time.time() - t0, 1)

    def __call__(self, embeds, seed: int, steps: int = SAMPLER["steps"], size: int = SAMPLER["height"]) -> tuple:
        import torch
        g = torch.Generator("cpu").manual_seed(int(seed))
        lat = torch.randn((1, self.pipe.transformer.in_channels, 2 * (size // 16), 2 * (size // 16)), generator=g,
                          dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()
        img = self.pipe(prompt=None, prompt_embeds=[embeds.to(self.device, _dtype(self.dtype))], height=size,
                        width=size, num_inference_steps=steps, guidance_scale=0.0, latents=lat.to(self.device),
                        output_type="np").images[0]
        torch.cuda.synchronize()
        arr = np.asarray(img, np.float32)
        return arr, {"seconds": round(time.time() - t0, 2), "finite": bool(np.isfinite(arr).all()),
                     "std": float(np.nan_to_num(arr).std()),
                     "peak_gpu_mib": round(torch.cuda.max_memory_allocated(self.device) / 2**20, 1)}


# -------------------------------------------------------------------------------------------- stage 3: keying
def key_background(rgb: np.ndarray, expected_rgb=None, t_lo: float = 0.10, t_hi: float = 0.22) -> tuple:
    """RGB float [0,1] image on a plain background -> (rgba, stats). Background colour = median of a 2 % border.
    alpha ramps from 0 to 1 as the colour distance to it goes from t_lo to t_hi; the object is the largest connected
    component of alpha > 0.5 plus every component at least 2 % of its size (separate parts of one object).
    Rejected (with the reason) when the border is not uniform, the object touches the frame, or the foreground
    fraction is outside [0.03, 0.85]."""
    from scipy import ndimage
    h, w, _ = rgb.shape
    b = max(4, int(0.02 * min(h, w)))
    border = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3), rgb[:, :b].reshape(-1, 3),
                             rgb[:, -b:].reshape(-1, 3)])
    bg = np.median(border, 0)
    spread = float(np.percentile(np.linalg.norm(border - bg, axis=1), 95))
    dist = np.linalg.norm(rgb - bg, axis=2)
    alpha = np.clip((dist - t_lo) / (t_hi - t_lo), 0, 1)
    lab, n = ndimage.label(alpha > 0.5)
    stats = {"background_measured": bg.round(4).tolist(), "background_expected": expected_rgb,
             "border_spread_p95": round(spread, 4), "components": int(n), "thresholds": [t_lo, t_hi],
             "method": "colour distance to the median border colour (no segmentation network)"}
    if spread > 0.12:
        raise Rejected(f"background not uniform (border colour spread p95 {spread:.3f} > 0.12)")
    if n == 0:
        raise Rejected("no foreground")
    sizes = ndimage.sum(np.ones_like(alpha), lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, 1 + np.nonzero(sizes >= 0.02 * sizes.max())[0])
    keep = ndimage.binary_dilation(keep, iterations=2)          # include the soft edge around kept parts
    alpha = alpha * keep
    hard = alpha > 0.5
    frac = float(hard.mean())
    edge = np.concatenate([hard[0], hard[-1], hard[:, 0], hard[:, -1]])
    stats.update({"foreground_fraction": round(frac, 4), "kept_components": int((sizes >= 0.02 * sizes.max()).sum()),
                  "touches_frame_fraction": round(float(edge.mean()), 4)})
    if not 0.03 <= frac <= 0.85:
        raise Rejected(f"foreground fraction {frac:.3f} outside [0.03, 0.85]")
    if edge.mean() > 0.01:
        raise Rejected(f"object touches the frame ({edge.mean():.3f} of the border is foreground)")
    a = alpha[..., None]
    fg = np.where(a > 1e-3, (rgb - (1 - a) * bg) / np.maximum(a, 1e-3), 0)   # remove the background's share (de-spill)
    return np.concatenate([np.clip(fg, 0, 1), alpha[..., None]], 2).astype(np.float32), stats


def u2net_path(cache: Path | None = None) -> Path:
    return Path(cache or DEFAULT_CACHE) / "u2net" / "u2net.onnx"


def fetch_u2net(cache: Path | None = None) -> dict:
    import urllib.request
    p = u2net_path(cache)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(U2NET["source"], p.with_suffix(".part"))
        p.with_suffix(".part").rename(p)
    try:
        return verify_file(p, "sha256", U2NET["sha256"], U2NET["bytes"])
    except ValueError:
        p.unlink(missing_ok=True)
        raise


_U2 = {}


def segment_u2net(rgb: np.ndarray, cache: Path | None = None) -> tuple:
    """rembg's U²-Net procedure (320² input, ImageNet normalisation after max-scaling, min-max normalised saliency
    resized back) -> (rgba, stats), with the same recorded gates as keying except background uniformity: rejected
    if the object touches the frame or the foreground fraction is outside [0.03, 0.85]."""
    import onnxruntime as ort
    from PIL import Image
    p = u2net_path(cache)
    if str(p) not in _U2:
        verify_file(p, "sha256", U2NET["sha256"], U2NET["bytes"])
        _U2[str(p)] = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    sess = _U2[str(p)]
    h, w, _ = rgb.shape
    im = np.asarray(Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)).resize(
        (320, 320), Image.Resampling.LANCZOS), np.float32)
    im = im / max(float(im.max()), 1e-6)
    im = (im - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    out = sess.run(None, {sess.get_inputs()[0].name: im.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    alpha = np.asarray(Image.fromarray((out * 255).astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS),
                       np.float32) / 255.0
    hard = alpha > 0.5
    frac = float(hard.mean())
    edge = np.concatenate([hard[0], hard[-1], hard[:, 0], hard[:, -1]])
    stats = {"method": "U²-Net salient object segmentation (rembg procedure), sha256 " + U2NET["sha256"][:16],
             "foreground_fraction": round(frac, 4), "touches_frame_fraction": round(float(edge.mean()), 4)}
    if not 0.03 <= frac <= 0.85:
        raise Rejected(f"foreground fraction {frac:.3f} outside [0.03, 0.85]")
    if edge.mean() > 0.01:
        raise Rejected(f"object touches the frame ({edge.mean():.3f} of the border is foreground)")
    return np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], 2).astype(np.float32), stats


def triposr_input(rgba: np.ndarray, size: int = TRIPOSR_CFG["input_size"],
                  ratio: float = TRIPOSR_CFG["foreground_ratio"]):
    """TripoSR's documented preprocessing: crop to the alpha bbox, pad square so the object fills ``ratio``,
    composite on 50 % grey, resize. -> PIL.Image (RGB)."""
    from PIL import Image
    ys, xs = np.nonzero(rgba[..., 3] > 0.5)
    crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(np.ceil(max(crop.shape[:2]) / ratio))
    canvas = np.zeros((side, side, 4), np.float32)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    a = canvas[..., 3:]
    rgb = canvas[..., :3] * a + 0.5 * (1 - a)
    return Image.fromarray((rgb * 255).round().astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS)


# ------------------------------------------------------------------------------------------ stage 4: TripoSR
def install_triposr_shims():
    """torchmcubes (CUDA extension) -> scikit-image marching cubes with torchmcubes' axis convention (D-020, unit-
    tested in M6); rembg -> a stub that refuses (keying above is the only background removal)."""
    import torch
    from skimage.measure import marching_cubes as sk_mc

    def marching_cubes(vol, thresh):
        v, f, _, _ = sk_mc(vol.detach().float().cpu().numpy(), level=float(thresh))
        v = np.ascontiguousarray(v[:, ::-1])            # torchmcubes returns (z, y, x); TripoSR reorders [2, 1, 0]
        return torch.from_numpy(v.astype(np.float32)), torch.from_numpy(f.astype(np.int64))
    sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=marching_cubes)

    def _no_rembg(*a, **k):
        raise RuntimeError("rembg is not used: zimage_triposr.key_background removes the background")
    sys.modules["rembg"] = types.SimpleNamespace(new_session=_no_rembg, remove=_no_rembg)


class Reconstructor:
    def __init__(self, device: str = "cuda", cache: Path | None = None):
        import torch
        code = REPO / "third_party" / "TripoSR"
        head = subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        if head.strip() != TRIPOSR["code_commit"]:
            raise RuntimeError(f"third_party/TripoSR is at {head.strip()!r}, not the pinned {TRIPOSR['code_commit']}")
        install_triposr_shims()
        if str(code) not in sys.path:
            sys.path.insert(0, str(code))
        from omegaconf import OmegaConf
        from tsr.system import TSR
        d = triposr_dir(cache)
        t0 = time.time()
        cfg = OmegaConf.load(d / "config.yaml")
        OmegaConf.resolve(cfg)
        self.model = TSR(cfg)
        self.model.load_state_dict(torch.load(d / "model.ckpt", map_location="cpu", mmap=True, weights_only=True),
                                   assign=True)
        self.model.renderer.set_chunk_size(TRIPOSR_CFG["chunk_size"])
        self.model.to(device).eval()
        self.device, self.load_s = device, round(time.time() - t0, 1)

    def __call__(self, image, resolution: int = TRIPOSR_CFG["mc_resolution"]) -> tuple:
        import torch
        from tsr.utils import to_gradio_3d_orientation
        t0 = time.time()
        with torch.no_grad():
            codes = self.model([image], device=self.device)
            mesh = self.model.extract_mesh(codes, True, resolution=resolution)[0]
        to_gradio_3d_orientation(mesh)                    # TripoSR z-up -> glTF/Berkelium +Y-up
        v = np.asarray(mesh.vertices, np.float64)
        f = np.asarray(mesh.faces, np.int64)
        col = np.asarray(mesh.visual.vertex_colors[:, :3], np.float32) / 255.0
        return v, f, col, {"seconds": round(time.time() - t0, 2), "mc_resolution": resolution}


def finalize_yup(v: np.ndarray, f: np.ndarray, col: np.ndarray | None):
    """Centre, scale the largest side to EXTENT, round to float32 FIRST, then drop zero-area faces on exactly those
    coordinates (the E-T3 P2 lesson): the returned mesh has no degenerate face by construction.
    -> (v float32, f, col, n_removed)."""
    from .geometry import clean_mesh
    lo, hi = v.min(0), v.max(0)
    v = ((v - (lo + hi) / 2) * (EXTENT / max(float((hi - lo).max()), 1e-9))).astype(np.float32).astype(np.float64)
    v2, f2, removed = clean_mesh(v, f)
    if col is not None:
        a = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
        col = np.clip(col[np.unique(f[a >= 1e-12])], 0, 1).astype(np.float32)
        if len(col) != len(v2):
            raise AssertionError("vertex colours out of step with cleaned vertices")
    return v2.astype(np.float32), f2, col, removed


def mesh_from_anchor(rgb: np.ndarray, background_rgb, recon: Reconstructor, segment: str = "u2net") -> dict:
    """Stages 3-4 for one anchor image -> {vertices, faces, colors, report, rgba, tsr_input}. ``segment``: "u2net"
    (v2, default) or "chroma_key" (v1, the pre-registered stage)."""
    from .geometry import validate_mesh
    rgba, kst = segment_u2net(rgb) if segment == "u2net" else key_background(rgb, background_rgb)
    img = triposr_input(rgba)
    v, f, col, rst = recon(img)
    rep = {"keying": kst, "triposr": rst, "raw_vertices": len(v), "raw_faces": len(f)}
    if len(f) == 0:
        rep.update(validate_mesh(None, None))
        return {"vertices": None, "faces": None, "colors": None, "report": rep, "rgba": rgba, "tsr_input": img}
    v2, f2, col2, removed = finalize_yup(v, f, col)
    rep.update(validate_mesh(v2, f2) if len(f2) else validate_mesh(None, None))
    rep["degenerate_faces_removed"] = removed
    return {"vertices": v2, "faces": f2, "colors": col2, "report": rep, "rgba": rgba, "tsr_input": img}


# ---------------------------------------------------------------------------------------------- the generator
def available(cache: Path | None = None) -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, ("needs a GPU: Z-Image-Turbo is a 6B-parameter diffusion transformer (fp16 ~12 GB); this host "
                       "has none. Benchmarked on Kaggle T4 (results/text3d/E-T3, device recorded per unit)")
    try:
        import diffusers
        if not hasattr(diffusers, "ZImagePipeline"):
            return False, f"diffusers {diffusers.__version__} has no ZImagePipeline (configs/environments/zimage-cuda.txt)"
    except ImportError:
        return False, "diffusers not installed (configs/environments/zimage-cuda.txt)"
    for ok, why in (_recorded(zimage_dir(cache), ZIMAGE["files"]), _recorded(triposr_dir(cache), TRIPOSR["files"])):
        if not ok:
            return False, why
    return True, ""


class ZImageTripoSRGenerator:
    """The Generator protocol over the four stages, staged in memory: encoder -> free -> transformer -> anchor ->
    free -> TripoSR. ``variant`` names an anchor LoRA (None = the base model); a LoRA is used only if it was
    promoted by E-T3 (NEURAL_RENDERING_DECISIONS.md)."""

    name = NAME

    def __init__(self, device: str = "cuda", dtype: str = "float16", cache: Path | None = None,
                 steps: int = SAMPLER["steps"], size: int = SAMPLER["height"]):
        self.device, self.dtype, self.cache, self.steps, self.size = device, dtype, cache, steps, size

    def available(self) -> tuple[bool, str]:
        return available(self.cache)

    @property
    def identity(self) -> dict:
        return {"generator": self.name, "model": "Z-Image-Turbo (f332072a) -> keyed anchor -> TripoSR (5b521936)",
                "pipeline": "anchor prompt template -> Z-Image-Turbo -> colour-key RGBA -> TripoSR -> clean mesh",
                "sampler": {**SAMPLER, "steps": self.steps, "height": self.size, "width": self.size},
                "dtype": self.dtype, "triposr": TRIPOSR_CFG, "weights": weight_identity(self.cache),
                "license": "Apache-2.0 (Z-Image-Turbo) + MIT (TripoSR)", "device": self.device}

    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict:
        import gc

        import torch

        def stage(frac, name):
            if cancel and cancel():
                raise InterruptedError("cancelled")
            if progress:
                progress(frac, name)
        ok, why = self.available()
        if not ok:
            from .generators import GeneratorUnavailable
            raise GeneratorUnavailable(why)
        t0 = time.time()
        ap = anchor_prompt(prompt)
        stage(0.0, "encoding")
        emb = encode_prompts([ap["text"]], self.device, self.cache)[0]
        stage(0.1, "anchor")
        gen = AnchorGenerator(self.device, self.dtype, self.cache)
        rgb, ast = gen(emb, seed, self.steps, self.size)
        del gen
        gc.collect()
        torch.cuda.empty_cache()
        rep = {"prompt": prompt, "seed": seed, "generator": self.name, "anchor_prompt": ap, "anchor": ast,
               "artifact_kind": "triangle_mesh", "units": "normalized_object",
               "frame": "TripoSR z-up -> +Y-up (to_gradio_3d_orientation), centred, max extent 1.6",
               "labels": "model-generated, normalized units, not measured geometry"}
        if not ast["finite"] or ast["std"] < 0.02:
            rep.update({"valid": False, "reason": f"anchor image non-finite or blank (std {ast['std']:.4f})"})
            return {"vertices": None, "faces": None, "colors": None, "report": rep, "anchor": rgb}
        stage(0.5, "reconstructing")
        recon = Reconstructor(self.device, self.cache)
        try:
            out = mesh_from_anchor(np.nan_to_num(rgb), ap["background_rgb"], recon)
        except Rejected as e:
            rep.update({"valid": False, "reason": f"anchor rejected: {e}"})
            return {"vertices": None, "faces": None, "colors": None, "report": rep, "anchor": rgb}
        finally:
            del recon
            gc.collect()
            torch.cuda.empty_cache()
        stage(0.95, "validating")
        rep.update(out["report"])
        rep["seconds"] = round(time.time() - t0, 1)
        return {"vertices": out["vertices"], "faces": out["faces"], "colors": out["colors"], "report": rep,
                "anchor": rgb, "anchor_rgba": out["rgba"]}
