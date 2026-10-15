"""Shap-E text-to-3D backend (OpenAI, MIT): an open-world pretrained 3D prior behind Berkelium's generator interface.

Pins
  code     openai/shap-e @ 50131012ee11c9d2617f3886c10f000d3c7a3b43 (third_party/shap-e, installed editable --no-deps)
  weights  text_cond.pt (text300M) and vector_decoder.pt, sha256 in WEIGHTS; checked by Shap-E's own downloader
           (URL_HASHES) on every load
  text     OpenAI CLIP ViT-L/14 @ d05afc436d78f1c48dc0dbf8e5980a9d471f35f6 (MIT), text tower only

CPU budget on a small host (7.8 GB RAM; processes above ~3.4 GB anonymous memory have been SIGTERMed here):
  * CLIP: the text tower's tensors are extracted ONCE (prepare_clip_text, its own process) bit-for-bit from the
    pinned ViT-L-14.pt into clip_vit_l14_text.pt next to the weights. At run time clip.model.CLIP's own text code is
    rebuilt from that file and the vision tower is replaced by a 1-parameter stub (CLIP.dtype reads
    visual.conv1.weight.dtype); text300M only encodes text. TorchScript is never used (torch.jit.load warns it may
    break on Python 3.14, and on this host it is SIGTERMed while loading ViT-L-14.pt).
  * checkpoints are loaded memory-mapped; the diffusion model and the mesh decoder are never resident together.
  * Shap-E's default cache is $CWD/shap_e_model_cache: every load passes cache_dir explicitly.
  * shap_e.util.notebooks (imports ipywidgets) is never imported: pan cameras and mesh decoding are local.
Output: Shap-E's STF mesh with vertex colours, converted from Shap-E's z-up frame to Berkelium's +Y-up frame and
normalised to the `normalized_object` extent of the native model. Model output, not measured geometry.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import os
import resource
import time
from pathlib import Path

import numpy as np

SHAPE_COMMIT = "50131012ee11c9d2617f3886c10f000d3c7a3b43"
CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
WEIGHTS = {"text_cond.pt": "e6b4fa599a7b3c3b16c222d5f5fe56f9db9289ff0b6575fbe5c11bc97106aad4",
           "vector_decoder.pt": "d7e7ebbfe3780499ae89b2da5e7c1354012dba5a6abfe295bed42f25c3be1b98",
           "ViT-L-14.pt": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"}
CLIP_TEXT_FILE = "clip_vit_l14_text.pt"
EXTENT = 1.6                      # largest bounding-box side, same as berkelium-procedural-shapes (shapes.EXTENT)
DEFAULT_CACHE = Path(os.environ.get("BERKELIUM_MODEL_CACHE",
                                    Path(__file__).resolve().parents[4] / "data" / "models")) / "shap_e"
NEEDED = ("text_cond.pt", "vector_decoder.pt", "text_cond_config.yaml", "vector_decoder_config.yaml",
          "diffusion_config.yaml", CLIP_TEXT_FILE)
STAGES = ("loading", "sampling", "decoding", "validating")


def available(cache_dir: Path | None = None) -> tuple[bool, str]:
    """(usable, reason). Never downloads: the weights must already be in the cache (scripts/setup_openworld.sh)."""
    try:
        import clip  # noqa: F401
        import shap_e  # noqa: F401
    except ImportError as e:
        return False, f"shap-e/clip not installed ({e.name}); see configs/environments/openworld-cpu.txt"
    cache = Path(cache_dir or DEFAULT_CACHE)
    missing = [n for n in NEEDED if not (cache / n).exists()]
    if missing:
        return False, f"weights missing in {cache}: {', '.join(missing)} (run scripts/setup_openworld.sh)"
    return True, ""


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def peak_rss_mib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


# ----------------------------------------------------------------------------------------------- CLIP text tower
_TEXT_PREFIXES = ("token_embedding.", "positional_embedding", "transformer.", "ln_final.", "text_projection",
                  "logit_scale")


class _VisualStub:
    """Stands in for CLIP's vision tower: CLIP.dtype reads visual.conv1.weight.dtype; nothing else is used for text.
    Any image call fails loudly instead of returning a wrong embedding."""

    @staticmethod
    def make(dtype):
        import torch

        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 1, 1, bias=False).to(dtype)
                self.input_resolution = 224

            def forward(self, *a, **k):
                raise RuntimeError("CLIP vision tower was dropped (text-only build); image conditioning unavailable")
        return Stub()


def _text_only_clip(state: dict, dtype):
    """Build clip.model.CLIP with the given text weights and a stub vision tower. Text hyperparameters are read from
    the state dict exactly as clip.model.build_model does; the throwaway vision tower is minimal (1 layer, width 64)."""
    import torch
    from clip.model import CLIP
    width = state["ln_final.weight"].shape[0]
    layers = len({k.split(".")[2] for k in state if k.startswith("transformer.resblocks")})
    m = CLIP(embed_dim=state["text_projection"].shape[1], image_resolution=224, vision_layers=1, vision_width=64,
             vision_patch_size=14, context_length=state["positional_embedding"].shape[0],
             vocab_size=state["token_embedding.weight"].shape[0], transformer_width=width,
             transformer_heads=width // 64, transformer_layers=layers)
    m.visual = _VisualStub.make(torch.float32)
    target = m.state_dict()
    state = {k: v for k, v in state.items() if k in target}      # drops TorchScript-only attributes (attn_mask)
    shape_bad = [k for k, v in state.items() if tuple(v.shape) != tuple(target[k].shape)]
    missing, unexpected = m.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    bad = [k for k in missing if not k.startswith("visual.")] + shape_bad
    if bad or unexpected:
        raise RuntimeError(f"CLIP text weights do not match the model: missing={bad} unexpected={unexpected}")
    m = m.to(dtype).eval()
    m.visual = _VisualStub.make(dtype)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


_STORAGE_DTYPES = {"HalfStorage": "float16", "FloatStorage": "float32", "DoubleStorage": "float64",
                   "LongStorage": "int64", "IntStorage": "int32", "BFloat16Storage": "bfloat16"}


def read_torchscript_tensors(path: Path, want) -> tuple[dict, dict]:
    """Named tensors of a TorchScript archive WITHOUT torch.jit.load (on this host jit.load of ViT-L-14.pt climbs past
    3.2 GB RSS while holding the GIL and is SIGTERMed by the memory watchdog).

    data.pkl is read by a restricted unpickler: only tensor-rebuild, storage-type, OrderedDict and __torch__ module
    classes resolve (anything else is refused), tensors become references, and only the storages of names for which
    ``want(name)`` is true are read from the (uncompressed) zip. -> ({name: tensor}, {name: (dtype, shape)} of all)."""
    import collections
    import pickle
    import zipfile

    import torch

    class _Module:
        def __setstate__(self, st):
            self.state = st

    class _Ref:
        def __init__(self, storage, offset, size, stride, *_rest):
            self.storage, self.offset, self.size, self.stride = storage, int(offset), tuple(size), tuple(stride)

    class _Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return _Ref
            if (mod, name) == ("collections", "OrderedDict"):
                return collections.OrderedDict
            if mod == "torch" and name in _STORAGE_DTYPES:
                return name
            if mod.startswith("__torch__."):
                return type(name, (_Module,), {})
            raise pickle.UnpicklingError(f"refused global {mod}.{name}")

        def persistent_load(self, pid):
            return pid                                   # ('storage', storage type, key, location, numel)

    refs: dict[str, _Ref] = {}

    def walk(o, prefix):
        st = getattr(o, "state", None)
        if not isinstance(st, dict):
            return
        for k, v in st.items():
            name = f"{prefix}{k}"
            if isinstance(v, _Ref):
                refs[name] = v
            elif isinstance(v, _Module):
                walk(v, name + ".")

    with zipfile.ZipFile(path) as z:
        root = z.namelist()[0].split("/")[0]
        with z.open(f"{root}/data.pkl") as fh:
            walk(_Unpickler(fh).load(), "")
        out, index = {}, {}
        for name, r in refs.items():
            dtype = getattr(torch, _STORAGE_DTYPES[r.storage[1]])
            index[name] = (str(dtype).removeprefix("torch."), r.size)
            if want(name):
                flat = torch.frombuffer(bytearray(z.read(f"{root}/data/{r.storage[2]}")), dtype=dtype)
                out[name] = flat.as_strided(r.size, r.stride, r.offset).clone()
    return out, index


def prepare_clip_text(cache_dir: Path | None = None) -> dict:
    """One-time, own process: ViT-L-14.pt (sha256-checked) -> its text-tower tensors, bit-for-bit in the checkpoint's
    own dtype -> CLIP_TEXT_FILE. Neither torch.jit.load nor clip.load is used: both materialise the whole model, and
    jit.load of this archive is SIGTERMed by this host's memory watchdog (see read_torchscript_tensors).
    Checks: the text tensors cover exactly the text-side keys of clip.model.CLIP (strict load, only visual.* missing,
    shapes equal), and the rebuilt tower ranks paraphrases above unrelated prompts."""
    import clip
    import torch
    cache = Path(cache_dir or DEFAULT_CACHE)
    src = cache / "ViT-L-14.pt"
    got = sha256_file(src)
    if got != WEIGHTS["ViT-L-14.pt"]:
        raise RuntimeError(f"{src} sha256 {got} != pinned {WEIGHTS['ViT-L-14.pt']}")
    state, index = read_torchscript_tensors(src, lambda n: n.startswith(_TEXT_PREFIXES))
    n_visual = sum(k.startswith("visual.") for k in index)
    text = _text_only_clip(state, torch.float32)     # strict on text keys; raises on any mismatch
    expect = {k for k in text.state_dict() if not k.startswith("visual.")}
    extra = sorted(set(state) - expect)              # e.g. TorchScript-only attributes such as attn_mask
    for k in extra:
        del state[k]
    probes = ["a red mug", "a red coffee cup", "a yellow school bus", "a wooden chair with four legs", "a chair"]
    with torch.no_grad():
        e = text.encode_text(clip.tokenize(probes)).float()
    e = e / e.norm(dim=-1, keepdim=True)
    sim = (e @ e.T).tolist()
    check = {"probes": probes, "cos_mug_cup": round(sim[0][1], 4), "cos_mug_bus": round(sim[0][2], 4),
             "cos_chair4_chair": round(sim[3][4], 4), "cos_chair4_bus": round(sim[3][2], 4)}
    if not (check["cos_mug_cup"] > check["cos_mug_bus"] and check["cos_chair4_chair"] > check["cos_chair4_bus"]):
        raise RuntimeError(f"text tower embeddings fail the paraphrase sanity check: {check}")
    out = cache / CLIP_TEXT_FILE
    tmp = out.with_suffix(".tmp")
    torch.save({"state_dict": state, "source": "ViT-L-14.pt", "source_sha256": WEIGHTS["ViT-L-14.pt"],
                "clip_commit": CLIP_COMMIT, "visual_tensors_dropped": n_visual, "ignored_attributes": extra,
                "check": check}, tmp)
    os.replace(tmp, out)
    return {"file": str(out), "sha256": sha256_file(out), "tensors": len(state), "visual_tensors_dropped": n_visual,
            "ignored_attributes": extra, "dtypes": sorted({str(v.dtype) for v in state.values()}),
            "params": int(sum(v.numel() for v in state.values())), "check": check, "peak_rss_mib": peak_rss_mib()}


@contextlib.contextmanager
def _patched_loaders(cache: Path, clip_dtype):
    """Within the block: torch.load memory-maps; clip.load returns the text-only CLIP built from CLIP_TEXT_FILE
    (Shap-E's ImageCLIP calls clip.load without a cache dir, which would download into the CWD)."""
    import clip
    import torch
    orig_load, orig_clip = torch.load, clip.load

    def load(f, *a, **k):
        k.setdefault("mmap", True)
        k.setdefault("weights_only", False)
        return orig_load(f, *a, **k)

    def clip_load(name, device="cpu", *a, **k):
        if name != "ViT-L/14":
            raise ValueError(f"text-only CLIP is prepared for ViT-L/14, not {name}")
        ck = orig_load(cache / CLIP_TEXT_FILE, map_location="cpu", mmap=True, weights_only=False)
        if ck.get("source_sha256") != WEIGHTS["ViT-L-14.pt"]:
            raise RuntimeError(f"{CLIP_TEXT_FILE} was not derived from the pinned ViT-L-14.pt")
        return _text_only_clip(ck["state_dict"], clip_dtype).to(device), None

    torch.load, clip.load = load, clip_load
    try:
        yield
    finally:
        torch.load, clip.load = orig_load, orig_clip


def load_shap_e_model(name: str, cache: Path, clip_dtype=None, device: str = "cpu"):
    """shap_e.models.download.load_model, but the checkpoint (hash-checked by Shap-E's fetch) is memory-mapped and,
    on CPU, assigned in place of the randomly initialised parameters (assign=True; strict) when every dtype and shape
    matches, so the weights stay file-backed instead of being copied into a second anonymous buffer. On a GPU device
    the parameters are copied to the device as usual (that path is UNVERIFIED: no GPU on the development host)."""
    import torch
    from shap_e.models.configs import model_from_config
    from shap_e.models.download import load_checkpoint, load_config
    with _patched_loaders(cache, clip_dtype or torch.float32):
        model = model_from_config(load_config(name, cache_dir=str(cache)), device=torch.device(device))
        sd = load_checkpoint(name, device=torch.device("cpu"), cache_dir=str(cache), progress=False)
    target = model.state_dict()
    assign = device == "cpu" and set(sd) == set(target) and all(
        sd[k].dtype == target[k].dtype and sd[k].shape == target[k].shape for k in sd)
    model.load_state_dict(sd, strict=True, assign=assign)
    del sd, target
    gc.collect()
    return model.eval()


# ------------------------------------------------------------------------------------------------ mesh decoding
def pan_cameras(size: int, device="cpu"):
    """Local copy of shap_e.util.notebooks.create_pan_cameras (that module imports ipywidgets)."""
    import torch
    from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
    origins, xs, ys, zs = [], [], [], []
    for theta in np.linspace(0, 2 * np.pi, num=20):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin), xs.append(x), ys.append(y), zs.append(z)
    t = lambda a: torch.from_numpy(np.stack(a, axis=0)).float().to(device)  # noqa: E731
    return DifferentiableCameraBatch(shape=(1, len(xs)), flat_camera=DifferentiableProjectiveCamera(
        origin=t(origins), x=t(xs), y=t(ys), z=t(zs), width=size, height=size, x_fov=0.7, y_fov=0.7))


def to_berkelium_frame(v: np.ndarray) -> np.ndarray:
    """Shap-E z-up (x, y, z) -> Berkelium +Y-up (x, z, -y) (a proper rotation), centred, largest side = EXTENT."""
    v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1).astype(np.float64)
    lo, hi = v.min(0), v.max(0)
    return (v - (lo + hi) / 2) * (EXTENT / max(float((hi - lo).max()), 1e-9))


def finalize_mesh(v: np.ndarray, f: np.ndarray, col: np.ndarray | None):
    """Frame conversion + Berkelium clean_mesh; colours follow exactly the vertices clean_mesh keeps.
    -> (v, f, col, n_degenerate_removed)."""
    from .geometry import clean_mesh
    # Round to the float32 coordinates that are returned FIRST, then clean in float64 on exactly those values (what
    # validate_mesh/trimesh measure): cleaning before the cast let a face of area ~1e-12 round to zero (E-T3 P2).
    v = to_berkelium_frame(v).astype(np.float32).astype(np.float64)
    v2, f2, removed = clean_mesh(v, f)
    if col is not None:
        a = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
        col = np.clip(col[np.unique(f[a >= 1e-12])], 0, 1).astype(np.float32)   # clean_mesh's own keep rule
        if len(col) != len(v2):
            raise AssertionError("vertex colours out of step with cleaned vertices")
    return v2.astype(np.float32), f2, col, removed


class ShapEGenerator:
    """text -> latent (text300M, Karras sampler) -> STF mesh (vector decoder). Deterministic for a given seed and
    thread count. ``cancel()`` is polled before every denoiser call and between stages; ``progress(frac, stage)``
    is reported per stage and per denoiser call."""

    name = "shap-e-text300m"

    def __init__(self, steps: int = 64, guidance_scale: float = 15.0, cache_dir: Path | None = None,
                 threads: int | None = None, grid_size: int | None = None, clip_dtype: str = "float32",
                 device: str = "cpu", fp16: bool | None = None):
        self.steps, self.guidance = int(steps), float(guidance_scale)
        self.cache = Path(cache_dir or DEFAULT_CACHE)
        self.threads, self.grid_size, self.clip_dtype, self.device = threads, grid_size, clip_dtype, device
        self.fp16 = (device != "cpu") if fp16 is None else bool(fp16)   # Shap-E's GPU default; CPU stays fp32

    def available(self) -> tuple[bool, str]:
        return available(self.cache)

    @property
    def identity(self) -> dict:
        return {"generator": self.name, "model": "Shap-E text300M + vector decoder (STF mesh)",
                "code": "openai/shap-e", "shap_e_commit": SHAPE_COMMIT, "clip_commit": CLIP_COMMIT,
                "weights_sha256": {k: WEIGHTS[k] for k in ("text_cond.pt", "vector_decoder.pt", "ViT-L-14.pt")},
                "conditioning": "CLIP ViT-L/14 text embedding (text tower only)",
                "license": "MIT (Shap-E code and weights); MIT (CLIP)",
                "sampler": {"karras_steps": self.steps, "guidance_scale": self.guidance, "sigma_min": 1e-3,
                            "sigma_max": 160, "s_churn": 0, "clip_denoised": True, "use_fp16": self.fp16},
                "decoder": {"rendering_mode": "stf", "grid_size": self.grid_size or "checkpoint default"},
                "clip_dtype": self.clip_dtype, "device": self.device,
                **({"gpu_path": "UNVERIFIED (never run on a GPU)"} if self.device != "cpu" else {}),
                "labels": "model-generated, normalized units, not measured geometry"}

    def latent(self, prompt: str, seed: int = 0, cancel=None, progress=None):
        import torch
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.diffusion.sample import sample_latents
        from shap_e.models.download import load_config
        if self.threads:
            torch.set_num_threads(self.threads)
        model = load_shap_e_model("text300M", self.cache, getattr(torch, self.clip_dtype), self.device)
        diffusion = diffusion_from_config(load_config("diffusion", cache_dir=str(self.cache)))
        calls = {"n": 0}
        expected = max(2 * self.steps - 1, 1)     # Heun (s_churn=0): 2 denoiser calls per step, 1 on the last; CFG
                                                  # doubles the batch inside each call, not the number of calls

        def hook(_m, _a):
            if cancel and cancel():
                raise InterruptedError("cancelled")
            calls["n"] += 1
            if progress:
                progress(min(calls["n"] / expected, 1.0), "sampling")
        h = model.register_forward_pre_hook(hook)
        torch.manual_seed(seed)
        try:
            with torch.no_grad():
                lat = sample_latents(batch_size=1, model=model, diffusion=diffusion, model_kwargs={"texts": [prompt]},
                                     guidance_scale=self.guidance, clip_denoised=True, use_fp16=self.fp16,
                                     use_karras=True, karras_steps=self.steps, sigma_min=1e-3, sigma_max=160,
                                     s_churn=0, progress=False)[0]
        finally:
            h.remove()
            del model
            gc.collect()
        return lat, calls["n"]

    def decode(self, lat):
        """Local equivalent of shap_e.util.notebooks.decode_latent_mesh -> (v, f, colours or None) in Shap-E's frame."""
        import torch
        from shap_e.util.collections import AttrDict
        dec = load_shap_e_model("decoder", self.cache, device=self.device)
        if self.grid_size:
            dec.renderer.grid_size = int(self.grid_size)
        try:
            with torch.no_grad():
                out = dec.renderer.render_views(AttrDict(cameras=pan_cameras(2, self.device)),
                                                params=dec.bottleneck_to_params(lat[None]),
                                                options=AttrDict(rendering_mode="stf", render_with_direction=False))
                tm = out.raw_meshes[0].tri_mesh()
        finally:
            del dec
            gc.collect()
        host = lambda a: a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)  # noqa: E731
        v = host(tm.verts).astype(np.float64)
        f = host(tm.faces).astype(np.int64)
        ch = tm.vertex_channels or {}
        col = np.stack([host(ch[c]) for c in "RGB"], 1).astype(np.float32) if all(c in ch for c in "RGB") else None
        return v, f, col

    def generate(self, prompt: str, seed: int = 0, cancel=None, progress=None) -> dict:
        """-> {vertices, faces, colors, report}, the same contract as text3d.generate.generate_mesh."""
        from .geometry import validate_mesh

        def stage(frac, name):
            if cancel and cancel():
                raise InterruptedError("cancelled")
            if progress:
                progress(frac, name)
        t0 = time.time()
        stage(0.0, "loading")
        lat, calls = self.latent(prompt, seed, cancel,
                                 (lambda fr, _n: progress(0.05 + 0.6 * fr, "sampling")) if progress else None)
        t1 = time.time()
        stage(0.65, "decoding")
        v, f, col = self.decode(lat)
        t2 = time.time()
        stage(0.95, "validating")
        rep = {"prompt": prompt, "seed": seed, "generator": self.name, "denoiser_calls": calls,
               "sample_s": round(t1 - t0, 1), "decode_s": round(t2 - t1, 1), "peak_rss_mib": peak_rss_mib(),
               "artifact_kind": "triangle_mesh", "units": "normalized_object",
               "frame": "shap-e z-up -> berkelium +Y-up (x, z, -y), centred, max extent 1.6",
               "labels": "model-generated, normalized units, not measured geometry",
               "raw_vertices": int(len(v)), "raw_faces": int(len(f))}
        if len(f) == 0:
            rep.update(validate_mesh(None, None))
            v2 = f2 = col2 = None
        else:
            v2, f2, col2, removed = finalize_mesh(v, f, col)
            rep.update(validate_mesh(v2, f2) if len(f2) else validate_mesh(None, None))
            rep["degenerate_faces_removed"] = removed
            if len(f2) == 0:
                v2 = f2 = col2 = None
        rep["seconds"] = round(time.time() - t0, 1)
        return {"vertices": v2, "faces": f2, "colors": col2, "report": rep}
