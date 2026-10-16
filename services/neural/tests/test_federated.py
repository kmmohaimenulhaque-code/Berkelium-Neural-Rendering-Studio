"""Phase I-V additions: zimage-triposr's CPU-checkable stages (anchor prompt, keying gates, TripoSR preprocessing,
float32-first clean), auto's evidence-ordered routing with recorded skips, adapter replay / separation-margin batches,
and the Kaggle runner's bundle hashing and pull verification. Model weights are never loaded here; generators are
fakes where a GPU model would be needed."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from berkelium_neural.text3d import generators as G
from berkelium_neural.text3d import zimage_triposr as Z
from berkelium_neural.text3d.geometry import validate_mesh

ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------------------------------- anchor prompt
def test_anchor_background_contrasts_with_named_colours():
    assert Z.anchor_prompt("a white porcelain teapot")["background"] == "bright green"
    assert Z.anchor_prompt("a teal plastic easter basket")["background"] == "bright magenta"
    a = Z.anchor_prompt("a large green box to the left of a purple box")
    assert a["background"] == "bright magenta" and a["colour_margin"] >= Z.MIN_BG_DISTANCE   # farthest from both
    assert "a large green box to the left of a purple box. A single complete object" in a["text"]
    with pytest.raises(Z.Rejected):           # nothing is far enough from all three named colours
        Z.anchor_prompt("a green and magenta and blue thing")


def test_every_e3t_prompt_gets_an_anchor_background():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_openworld_benchmark as rob
    cfg, prompts = rob.load_cfg()
    units, _ = rob.plan_units(cfg, prompts, ["B1", "B2", "B3_all", "B4_single", "B4_planner"], gen=Z.NAME)
    assert len(units) == 56 and all(Z.anchor_prompt(u["prompt"])["colour_margin"] >= 0.6 for u in units)


# ------------------------------------------------------------------------------------------------- keying
def _anchor(bg=(0, 177, 64), touch=False, noise=0.01):
    h = w = 128
    img = np.zeros((h, w, 3), np.float32) + np.array(bg, np.float32) / 255
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - 64, xx - 64)
    img[(r < 35) & (r > 10)] = (0.8, 0.1, 0.1)          # a ring: its hole must stay background
    if touch:
        img[55:75, :40] = (0.8, 0.1, 0.1)                # an arm of the object reaching the left edge
    return np.clip(img + np.random.default_rng(0).normal(0, noise, img.shape), 0, 1).astype(np.float32)


def test_keying_keeps_holes_and_removes_background():
    rgba, st = Z.key_background(_anchor(), [0, 177, 64])
    assert rgba[64, 64, 3] == 0 and rgba[64, 64 + 22, 3] == 1 and 0.05 < st["foreground_fraction"] < 0.4
    assert np.allclose(rgba[64, 86, :3], (0.8, 0.1, 0.1), atol=0.05)          # de-spilled object colour


@pytest.mark.parametrize("kind,match", [("touch", "touches the frame"), ("noisy", "not uniform")])
def test_keying_rejects_with_a_reason(kind, match):
    img = _anchor(touch=kind == "touch", noise=0.2 if kind == "noisy" else 0.01)
    with pytest.raises(Z.Rejected, match=match):
        Z.key_background(img)


def test_triposr_input_is_grey_square_with_object_fraction():
    rgba, _ = Z.key_background(_anchor())
    im = np.asarray(Z.triposr_input(rgba, size=256, ratio=0.85), np.float32) / 255
    assert im.shape == (256, 256, 3) and np.allclose(im[0, 0], 0.5, atol=0.01)
    obj = np.abs(im - 0.5).sum(-1) > 0.1
    ys, xs = np.nonzero(obj)
    assert 0.80 < (xs.max() - xs.min() + 1) / 256 < 0.90                         # the ring fills 85 % of the side


def test_finalize_yup_is_clean_by_construction():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(60, 3)) * np.array([1.0, 3.0, 0.5])
    f = rng.integers(0, 60, (100, 3))
    f[:3] = [[2, 2, 5], [7, 8, 7], [9, 9, 9]]                                 # zero-area faces
    col = rng.random((60, 3)).astype(np.float32)
    v2, f2, c2, removed = Z.finalize_yup(v, f, col)
    rep = validate_mesh(v2, f2)
    assert removed >= 3 and rep["degenerate_faces"] == 0 and v2.dtype == np.float32
    assert np.isclose(np.ptp(v2, 0).max(), Z.EXTENT, atol=1e-5) and np.allclose((v2.min(0) + v2.max(0)) / 2, 0,
                                                                                 atol=1e-5)
    assert len(c2) == len(v2)


def test_pins_cover_every_pipeline_file_and_hash_kinds_are_known():
    assert {p.split("/")[0] for p in Z.ZIMAGE["files"]} == {"model_index.json", "scheduler", "text_encoder",
                                                            "tokenizer", "transformer", "vae"}
    assert all(k in ("sha256", "git_sha1") and len(h) == (64 if k == "sha256" else 40)
               for k, h, _ in [*Z.ZIMAGE["files"].values(), *Z.TRIPOSR["files"].values()])
    blob = b"hello\n"
    p = ROOT / "data" / "_git_blob_probe.txt"
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(blob)
    try:   # `git hash-object` of "hello\n"
        assert Z.git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"
        assert Z.verify_file(p, "git_sha1", "ce013625030ba8dba906f756967f9e9ca394464a", 6)["bytes"] == 6
        with pytest.raises(ValueError):
            Z.verify_file(p, "sha256", "0" * 64, 6)
    finally:
        p.unlink()


# ----------------------------------------------------------------------------------------------- routing
class _Fake:
    def __init__(self, name, ok, why=""):
        self.name, self.ok, self.why, self.calls = name, ok, why, 0

    def available(self):
        return self.ok, self.why

    @property
    def identity(self):
        return {"generator": self.name}

    def generate(self, prompt, seed=0, cancel=None, progress=None):
        self.calls += 1
        return {"vertices": np.zeros((3, 3), np.float32), "faces": np.array([[0, 1, 2]]), "colors": None,
                "report": {"valid": True}}


def test_auto_takes_first_available_in_evidence_order_and_records_skips(monkeypatch):
    monkeypatch.setattr(G, "OPEN_WORLD_RANKING", (G.ZIMAGE_TRIPOSR, G.SHAP_E))
    z, s = _Fake(G.ZIMAGE_TRIPOSR, False, "GPU-gated: needs a GPU"), _Fake(G.SHAP_E, True)
    out = G.AutoGenerator(open_world={z.name: z, s.name: s}).generate("a wooden chair")
    r = out["report"]["routing"]
    assert r["selected"] == G.SHAP_E and r["generator"] == G.ZIMAGE_TRIPOSR and s.calls == 1 and z.calls == 0
    assert r["skipped_unavailable"] == [{"generator": G.ZIMAGE_TRIPOSR, "reason": "GPU-gated: needs a GPU"}]


def test_auto_fails_with_every_reason_when_nothing_is_available(monkeypatch):
    monkeypatch.setattr(G, "OPEN_WORLD_RANKING", (G.ZIMAGE_TRIPOSR, G.SHAP_E))
    gens = {n: _Fake(n, False, f"{n} missing") for n in (G.ZIMAGE_TRIPOSR, G.SHAP_E)}
    with pytest.raises(G.GeneratorUnavailable,
                       match="zimage-triposr missing; shap-e-text300m: shap-e-text300m missing"):
        G.AutoGenerator(open_world=gens).generate("a wooden chair")


def test_zimage_unavailable_on_a_cpu_host_names_the_reason():
    ok, why = G.ZImageTripoSR().available()
    assert not ok and why.startswith("GPU-gated")
    with pytest.raises(G.GeneratorUnavailable):
        G.ZImageTripoSR().generate("a teapot")


# ------------------------------------------------------------------------------------- adapter training data
def test_replay_batch_mix_and_gap_points():
    from berkelium_neural.text3d.shapes import Composition, Part, sdf
    from berkelium_neural.text3d.train import make_batch
    a = Composition("next_to", (Part("capsule", "red", "", (0.2, 0.3, 0.2)), Part("box", "blue", "",
                                                                                  (0.3, 0.3, 0.3))), 0)
    b = Composition("on_top_of", (Part("sphere", "green", "", (0.3, 0.3, 0.3)), Part("box", "white", "",
                                                                                     (0.3, 0.3, 0.3))), 0)
    cfg = {"seed": 0, "batch_assets": 4, "points_per_asset": 64, "replay": {"fraction": 0.5},
           "separation_margin": {"weight": 1.0, "margin": 0.02, "points_per_asset": 32}}
    texts, pts, gt, col, (gp, gd, gw) = make_batch([(a, "own")], 3, cfg, replay=[(b, "replayed")])
    assert len(texts) == 4 and pts.shape == (4, 64, 3) and gp.shape == (4, 32, 3)
    w = gw.numpy()
    assert w[:2].sum() > 0 and w[2:].sum() == 0              # gap points only for the side-by-side composition
    q = gp.numpy()[0][w[0] > 0]
    assert np.all(sdf(a, q) > 0.005) and np.allclose(gd.numpy()[0][w[0] > 0], sdf(a, q), atol=1e-6)
    cfg_old = {"seed": 0, "batch_assets": 4, "points_per_asset": 64}
    assert len(make_batch([(a, "own")], 3, cfg_old)) == 4    # no new keys -> the original 4-tuple batch


# ------------------------------------------------------------------------------------------- Kaggle runner
def test_runner_bundle_is_reproducible_and_pull_verifies(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import kaggle_gpu_runner as K
    spec = {"name": "t", "include": ["services/neural/berkelium_neural/text3d"], "steps": [], "outputs": []}
    b1, files = K.make_bundle(spec)
    b2, _ = K.make_bundle(spec)
    assert b1 == b2 and "services/neural/berkelium_neural/text3d/zimage_triposr.py" in files
    # a fake kernel output: one good file, then a tampered one
    monkeypatch.setattr(K, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(K, "ROOT", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    st = {"run_id": "r1", "kernel": "u/k"}

    def fake_output(files_ok=True):
        def run(*args, **kw):
            stage = Path(args[args.index("-p") + 1]) / "out"
            (stage / "files" / "results").mkdir(parents=True)
            data = b"payload"
            (stage / "files" / "results" / "a.txt").write_bytes(data if files_ok else b"tampered")
            (stage / "manifest.json").write_text(json.dumps({"results/a.txt": {
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}))
            (stage / "receipt.json").write_text(json.dumps({"status": "completed", "steps": []}))
            return ""
        return run
    monkeypatch.setattr(K, "kaggle", fake_output(True))
    K.pull(spec, dict(st), admit=True)
    assert (tmp_path / "repo" / "results" / "a.txt").read_bytes() == b"payload"
    monkeypatch.setattr(K, "kaggle", fake_output(False))
    with pytest.raises(K.RunnerError, match="verification failed"):
        K.pull(spec, dict(st), admit=True)
