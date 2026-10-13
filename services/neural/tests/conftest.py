import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]          # repository root
NEURAL = ROOT / "services" / "neural"
sys.path.insert(0, str(NEURAL))
sys.path.insert(0, str(ROOT / "scripts"))

GSPLAT_SRC = Path(os.environ.get("BERKELIUM_GSPLAT_SRC", ROOT / "third_party" / "amd-gsplat"))


@pytest.fixture(scope="session")
def gsplat_torch_impl():
    """Pinned AMD gsplat pure-PyTorch reference functions (CPU-runnable). Skips if the pin is not checked out."""
    if not (GSPLAT_SRC / "gsplat" / "cuda" / "_torch_impl.py").exists():
        pytest.skip("AMD gsplat source not found; run scripts/setup_cpu.sh")
    sys.path.insert(0, str(GSPLAT_SRC))
    from gsplat.cuda import _torch_impl
    return _torch_impl


@pytest.fixture(scope="session")
def hipcpu_lib(tmp_path_factory):
    from berkelium_neural.render.berkelium_ext import build_hipcpu_library
    from berkelium_neural.render.interface import RendererError
    try:
        return build_hipcpu_library(tmp_path_factory.mktemp("bkcache"))
    except RendererError as e:
        pytest.skip(str(e))


TINY_DATASET = {
    "name": "tiny-test", "seed": 7, "license": "MIT (Berkelium-owned procedural assets)", "permissions": "test",
    "adapter_concept": "capsule", "adapter_test_partners": ["cone", "torus", "capsule"], "n_test_families": 4,
    "n_val_families": 2, "variants": {"train": 1, "val": 1, "test": 1, "adapter_train": 1, "adapter_test": 1},
    "pilot_images_per_train_family": 0, "gt_mesh_res": 24, "image_size": 16, "camera_radius": 3.2, "train_views": 2,
    "train_elevation_deg": 20.0, "eval_views": 2, "eval_elevation_deg": 35.0, "camera_convention": "opencv",
    "split_policy": "family-level"}


def tiny_train_cfg(root, out, **kw):
    cfg = {"kind": "base", "dataset": str(root), "train_splits": ["train"], "steps": 10, "batch_assets": 4,
           "points_per_asset": 128, "lr": 1e-3, "seed": 3, "threads": 1, "checkpoint_every": 5, "log_every": 5,
           "model": {"d": 16, "z_dim": 16, "hidden": 32}, "out_dir": str(out)}
    cfg.update(kw)
    return cfg


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """Tiny dataset + 10-step base model + 10-step LoRA adapter (CPU, seconds)."""
    from berkelium_neural.text3d.dataset import build
    from berkelium_neural.text3d.train import train
    d = tmp_path_factory.mktemp("tiny")
    card = build(TINY_DATASET, d / "data", log=lambda *a: None)
    base = train(tiny_train_cfg(d / "data", d / "base"), log=lambda *a: None)
    lora = train(tiny_train_cfg(d / "data", d / "lora", kind="lora", train_splits=["adapter_train"],
                                base_checkpoint=base["artifact"], lora_r=2, lora_alpha=4, new_tokens=["capsule"],
                                model=None), log=lambda *a: None)
    return {"root": d, "data": d / "data", "card": card, "base": base, "lora": lora}
