"""Text-to-3D pathway: owned dataset (splits, leakage, provenance), training determinism/resume, LoRA semantics,
generation -> valid mesh, evaluation protocol, metrics sanity."""

import json

import numpy as np
import pytest
import torch
from conftest import tiny_train_cfg

from berkelium_neural.text3d import shapes
from berkelium_neural.text3d.dataset import check_no_leakage, composition_of, load_manifest
from berkelium_neural.text3d.evaluate import ModelGenerator, RetrievalBaseline, evaluate
from berkelium_neural.text3d.generate import generate_mesh, mesh_to_gaussians, mesh_to_glb
from berkelium_neural.text3d.geometry import geometry_metrics, grid_points, mesh_from_sdf_grid, validate_mesh
from berkelium_neural.text3d.model import build_vocab, tokenize
from berkelium_neural.text3d.train import load_model, train


def test_primitive_sdfs_are_exact_distance_fields():
    rng = np.random.default_rng(0)
    for sh in shapes.SHAPES:
        p = shapes.Part(sh, "red", "", shapes._part_dims(sh, "", rng))
        q = rng.uniform(-1, 1, (4000, 3))
        e = 1e-4
        g = np.stack([(shapes._sdf_part(p, q + e * np.eye(3)[k]) - shapes._sdf_part(p, q - e * np.eye(3)[k])) / (2 * e)
                      for k in range(3)], 1)
        d = shapes._sdf_part(p, q)
        gn = np.linalg.norm(g, axis=1)[np.abs(d) > 0.01]
        assert np.quantile(np.abs(gn - 1), 0.99) < 1e-3, sh       # eikonal |grad d| = 1


def test_dataset_splits_provenance_and_leakage(tiny_run):
    recs = load_manifest(tiny_run["data"])
    rep = check_no_leakage(recs)
    assert rep["family_overlap"] == 0 and rep["asset_overlap"] == 0
    assert not any("capsule" in r.family for r in recs if r.split in ("train", "val", "test"))
    for r in recs:
        assert r.license.startswith("MIT") and r.caption_provenance.startswith("template:")
        c = composition_of(r)
        assert c.asset_sha256() == r.asset_sha256 and c.family == r.family
        if r.split in ("val", "test", "adapter_test"):
            assert r.files["gt_mesh"] and len(r.cameras) == 2
    card = tiny_run["card"]
    assert set(card["split_hashes"]) == {"train", "val", "test", "adapter_train", "adapter_test"}


def test_tokenizer_flags_unknown_words():
    v = build_vocab()
    ids, unk = tokenize("a shiny red sphere", v)
    assert unk == ["shiny"] and ids[1] == 1


def test_training_is_deterministic_and_resumable(tiny_run, tmp_path):
    full = train(tiny_train_cfg(tiny_run["data"], tmp_path / "full"), log=lambda *a: None)
    part = train(tiny_train_cfg(tiny_run["data"], tmp_path / "part", max_steps_this_run=5), log=lambda *a: None)
    assert part["step"] == 5 and not part["complete"]
    rest = train(tiny_train_cfg(tiny_run["data"], tmp_path / "part"), log=lambda *a: None)
    assert rest["complete"]
    a = torch.load(full["artifact"], weights_only=False)["model"]
    b = torch.load(rest["artifact"], weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)              # resume == uninterrupted, bitwise
    with pytest.raises(ValueError):                              # changed config cannot hijack a checkpoint
        train(tiny_train_cfg(tiny_run["data"], tmp_path / "part", lr=5e-3, steps=20), log=lambda *a: None)
    rc = json.loads((tmp_path / "full" / "receipt.json").read_text())["receipt"]
    assert rc["gpu_hours"] == 0.0 and rc["peak_rss_mb"] > 0 and rc["dataset_manifest_sha256"]


def test_lora_adapter_freezes_base_and_binds_to_it(tiny_run):
    base = load_model(tiny_run["base"]["artifact"])
    ad = load_model(tiny_run["base"]["artifact"], tiny_run["lora"]["artifact"])
    sd_b = base.state_dict()
    for k, v in ad.state_dict().items():                          # frozen base weights identical
        kb = k.replace(".base.", ".")
        if kb in sd_b and not k.endswith((".A", ".B")):
            assert torch.equal(v, sd_b[kb]), k
    delta = ad.emb_delta.detach()
    cap = ad.vocab.index("capsule")
    assert delta[cap].abs().sum() > 0
    assert delta[torch.arange(len(ad.vocab)) != cap].abs().sum() == 0   # only the new token's row moves
    st = torch.load(tiny_run["lora"]["artifact"], weights_only=False)
    st["base_checkpoint_sha256"] = "0" * 64
    bad = tiny_run["root"] / "bad_adapter.pt"
    torch.save(st, bad)
    with pytest.raises(ValueError):
        load_model(tiny_run["base"]["artifact"], bad)


def test_generation_produces_real_mesh_and_gaussians():
    """A mesh from the exact SDF path (no model) proves the meshing/export/Gaussian chain is real 3D."""
    c = shapes.sample_composition(np.random.default_rng(1), "on_top_of", ("sphere", "box"))
    vol = shapes.sdf(c, grid_points(32)).reshape(32, 32, 32)
    v, f = mesh_from_sdf_grid(vol)
    rep = validate_mesh(v, f)
    assert rep["valid"] and rep["watertight"] and rep["edge_manifold"] and rep["components"] == 1
    import io

    import trimesh
    m = trimesh.load(io.BytesIO(mesh_to_glb(v, f, shapes.colors_at(c, v))), file_type="glb", force="mesh")
    assert len(m.faces) == len(f)
    g = mesh_to_gaussians(v, f, shapes.colors_at(c, v), n=500)
    assert g.validate()["n"] == 500


def test_model_generation_and_eval_protocol(tiny_run):
    m = load_model(tiny_run["base"]["artifact"])
    out = generate_mesh(m, "a red sphere", resolution=20)
    assert out["report"]["artifact_kind"] == "triangle_mesh" and "valid" in out["report"]
    res = evaluate(ModelGenerator(m, 20, "tiny"), tiny_run["data"], "test", log=None)
    agg = res["aggregate"]
    assert agg["n_records"] == 4 and agg["lpips"]["status"] == "not_run"
    for r in res["records"]:                                     # failures are recorded, never zero-filled
        if not r["success"]:
            assert r["chamfer_l1"] is None or r["status"] != "no_prediction"
    base = evaluate(RetrievalBaseline(load_manifest(tiny_run["data"], {"train"}), 24), tiny_run["data"], "test",
                    log=None)
    assert base["aggregate"]["success_rate"] == 1.0


def test_geometry_metric_sanity():
    c = shapes.sample_composition(np.random.default_rng(2), "single", ("sphere",))
    v, f = mesh_from_sdf_grid(shapes.sdf(c, grid_points(48)).reshape(48, 48, 48))
    same = geometry_metrics(v, f, v, f, n=5000)
    shifted = geometry_metrics(v + np.array([0.2, 0, 0], np.float32), f, v, f, n=5000)
    assert same["chamfer_l1"] < 0.03 and same["fscore@0.05"] > 0.98          # 5k samples: spacing ~0.04
    assert shifted["chamfer_l1"] > 5 * same["chamfer_l1"] and shifted["fscore@0.02"] < same["fscore@0.02"]
    assert geometry_metrics(None, None, v, f)["status"] == "no_prediction"


def test_ml_interfaces_have_real_implementations(tiny_run, tmp_path):
    from berkelium_neural.text3d.interfaces import BerkeliumTextShape, LoRAAdapter, Text3DModel, TrainingAdapter
    from berkelium_neural.text3d.train import file_sha256
    m = BerkeliumTextShape(tiny_run["base"]["artifact"])
    assert isinstance(m, Text3DModel) and m.capabilities["text_to_3d"] and not m.capabilities["image_to_3d"]
    assert m.encode_text(["a red sphere", "a box"]).shape[0] == 2
    out = m.generate("a red sphere", resolution=20)
    assert out.report["artifact_kind"] == "triangle_mesh"
    ad = LoRAAdapter(r=2, alpha=4, new_tokens=("capsule",))
    assert isinstance(ad, TrainingAdapter)
    model = load_model(tiny_run["base"]["artifact"])
    params = ad.attach(model)
    assert params and all(p.requires_grad for p in params)
    ad.save(model, tmp_path / "a.pt", file_sha256(tiny_run["base"]["artifact"]))
    assert load_model(tiny_run["base"]["artifact"], tmp_path / "a.pt").emb_delta is not None


def test_lora_is_applied_in_eval_fast_path_and_train_path(tiny_run):
    """Regression: nn.TransformerEncoderLayer's eval fast path reads linear.weight directly; LoRA must still apply."""
    ad = load_model(tiny_run["base"]["artifact"], tiny_run["lora"]["artifact"])
    base = load_model(tiny_run["base"]["artifact"])
    prompts = ["a red capsule on top of a blue box", "a green sphere"]
    with torch.no_grad():
        z_eval = ad.encode(prompts)
        ad.train()
        z_train = ad.encode(prompts)
        ad.eval()
        z_base = base.encode(prompts)
    assert torch.allclose(z_eval, z_train, atol=1e-5)          # fused fast path == module forward
    assert not torch.allclose(z_eval, z_base, atol=1e-4)       # the adapter actually changes the output
    lin = ad.decoder.l0
    x = torch.randn(5, lin.in_features)
    assert torch.allclose(lin(x), torch.nn.functional.linear(x, lin.weight, lin.bias), atol=1e-5)
    out = generate_mesh(ad, prompts[0], resolution=20)        # full inference path with the adapter
    assert out["report"]["artifact_kind"] == "triangle_mesh"
