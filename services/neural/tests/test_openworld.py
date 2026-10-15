"""Open-world generator layer (E-T3): auto routing, the two-object planner, Shap-E frame/colour handling, availability
reasons and provenance. Mocks appear only here (fake generators, a stubbed Shap-E sampler); never in benchmarks/E2E."""

import os
import pickle
import zipfile

import numpy as np
import pytest

from berkelium_neural.text3d import generators as G
from berkelium_neural.text3d import planner as P
from berkelium_neural.text3d import shap_e_backend as S
from berkelium_neural.text3d.model import build_vocab


def box(lo=(-0.5, -0.5, -0.5), hi=(0.5, 0.5, 0.5)):
    import trimesh
    m = trimesh.creation.box(bounds=np.array([lo, hi]))
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)


# ----------------------------------------------------------------------------------------------------- routing
@pytest.mark.parametrize("prompt,expected,unknown", [
    ("a large red box on top of a green sphere", G.NATIVE, []),
    ("A cyan torus to the left of a red sphere.", G.NATIVE, []),
    ("a sphere and a box", G.NATIVE, []),                          # "and" is a function word
    ("the sphere", G.NATIVE, []),
    ("a wooden chair with four legs", G.SHAP_E, ["chair", "four", "legs", "wooden"]),
    ("a red mug", G.SHAP_E, ["mug"]),
    ("a small blue cube", G.SHAP_E, ["cube"]),                     # no synonym table: "cube" is not "box"
    ("an elephant", G.SHAP_E, ["elephant"]),
])
def test_route_native_iff_no_unknown_content_word(prompt, expected, unknown):
    d = G.route(prompt, build_vocab())
    assert d["generator"] == expected and d["unknown_words"] == unknown


def test_route_uses_the_shipped_checkpoint_vocab():
    auto = G.AutoGenerator()
    assert auto.native.vocab() == build_vocab()
    assert auto.decide("a red mug")["generator"] == G.SHAP_E


# ----------------------------------------------------------------------------------------------------- planner
@pytest.mark.parametrize("prompt,relation,a,b", [
    ("a red box on top of a green sphere", "on_top_of", "a red box", "a green sphere"),
    ("a small red cone stacked on a large blue box", "on_top_of", "a small red cone", "a large blue box"),
    ("a blue box with a yellow cone on top", "on_top_of", "a yellow cone", "a blue box"),
    ("a mug on a table", "on_top_of", "a mug", "a table"),
    ("a cyan torus to the left of a red sphere", "next_to", "a cyan torus", "a red sphere"),
    ("a red sphere to the right of a blue box", "next_to", "a blue box", "a red sphere"),
    ("a red box beside a blue sphere", "next_to", "a red box", "a blue sphere"),
    ("a red box and a blue sphere", "next_to", "a red box", "a blue sphere"),
])
def test_planner_parses_supported_relations(prompt, relation, a, b):
    plan, _ = P.parse(prompt)
    assert (plan.relation, plan.a, plan.b) == (relation, a, b)


@pytest.mark.parametrize("prompt", ["a black and white cow", "a wooden chair with four legs",
                                    "a cup on a saucer next to a spoon", "red box on blue sphere", "a red mug"])
def test_planner_refuses_single_nested_or_bare_phrases(prompt):
    assert P.parse(prompt)[0] is None


def test_place_on_top_of_touches_and_aligns():
    va, _ = box((-1, -1, -1), (1, 1, 1))
    vb, _ = box((0, 0, 0), (3, 1, 2))
    Ta, Tb = P.place(P.parse("a red box on top of a large blue box")[0], va, vb)
    a, b = P._apply(Ta, va), P._apply(Tb, vb)
    assert a.min(0)[1] == pytest.approx(b.max(0)[1], abs=1e-6)                       # touching
    for ax in (0, 2):
        assert (a.min(0)[ax] + a.max(0)[ax]) / 2 == pytest.approx((b.min(0)[ax] + b.max(0)[ax]) / 2, abs=1e-6)
    allv = np.concatenate([a, b])
    assert (allv.max(0) - allv.min(0)).max() == pytest.approx(P.EXTENT)
    assert (allv.max(0) + allv.min(0)) == pytest.approx(np.zeros(3), abs=1e-6)
    # relative size from the size words: large (1.3) vs default (1.0)
    assert (b.max(0) - b.min(0)).max() / (a.max(0) - a.min(0)).max() == pytest.approx(1.3)


def test_place_next_to_gap_and_order():
    va, _ = box()
    vb, _ = box()
    Ta, Tb = P.place(P.parse("a small red box next to a blue box")[0], va, vb)
    a, b = P._apply(Ta, va), P._apply(Tb, vb)
    s = Tb[0, 0] * 1.0                                       # scene scale of a default part (unit extent box)
    assert a.max(0)[0] < b.min(0)[0]                          # A left of B
    assert (b.min(0)[0] - a.max(0)[0]) / s == pytest.approx(P.GAP)
    assert (a.max(0) - a.min(0)).max() / s == pytest.approx(0.7)
    assert (a.min(0)[1] + a.max(0)[1]) == pytest.approx(b.min(0)[1] + b.max(0)[1], abs=1e-6)


class FakeGen:
    def __init__(self, name="fake"):
        self.name, self.calls = name, []

    identity = property(lambda self: {"generator": self.name})

    def available(self):
        return True, ""

    def generate(self, prompt, seed=0, cancel=None, progress=None):
        self.calls.append(prompt)
        v, f = box()
        return {"vertices": v * 1.6, "faces": f, "colors": np.full((len(v), 3), 0.5, np.float32),
                "report": {"valid": True, "prompt": prompt}}


def test_generate_scene_parts_are_separate_validated_and_cached():
    g, cache = FakeGen(), {}
    out, _ = P.generate_scene("a red box next to a red box", lambda t: g, cache=cache)
    assert out["status"] == "ok" and g.calls == ["a red box"]            # identical part generated once
    assert out["scene"]["report"]["components"] == 2 and out["scene"]["report"]["valid"]
    assert all(p["report"]["valid"] for p in out["parts"])
    T = np.array(out["parts"][0]["transform"])
    v0 = FakeGen().generate("x")["vertices"]
    assert np.allclose(P._apply(T, v0.astype(np.float64)), out["parts"][0]["vertices"], atol=1e-6)
    assert P.generate_scene("a red mug", lambda t: g)[0] is None


# ------------------------------------------------------------------------------------------------ Shap-E frame
def test_shap_e_frame_is_a_proper_rotation_to_y_up_and_normalised():
    v = np.array([[0, 0, 0], [0, 0, 2.0], [1.0, 0, 0], [0, 0.5, 0]])     # z-up: (0,0,2) is "up"
    out = S.to_berkelium_frame(v)
    up = out[1] - out[0]
    assert up[1] > 0 and abs(up[0]) < 1e-12 and abs(up[2]) < 1e-12       # Shap-E +Z -> Berkelium +Y
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert np.linalg.det(R) == pytest.approx(1.0)                         # no mirror
    assert (out.max(0) - out.min(0)).max() == pytest.approx(S.EXTENT)
    assert (out.max(0) + out.min(0)) == pytest.approx(np.zeros(3))


def test_finalize_mesh_keeps_colours_aligned_with_cleaned_vertices():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2], [5, 5, 5]], float)
    f = np.array([[0, 1, 2], [0, 1, 3], [4, 4, 1]])                     # last face is degenerate
    col = np.linspace(0, 1, 18).reshape(6, 3).astype(np.float32)          # unique colour per vertex
    v2, f2, c2, removed = S.finalize_mesh(v, f, col)
    assert removed == 1 and len(v2) == 4 == len(c2)
    conv = S.to_berkelium_frame(v)
    for i, p in enumerate(v2):                                            # each kept vertex keeps its own colour
        j = int(np.argmin(np.linalg.norm(conv - p, axis=1)))
        assert np.allclose(c2[i], col[j])


def test_finalize_mesh_cleans_faces_that_degenerate_only_after_float32_rounding():
    from berkelium_neural.text3d.geometry import validate_mesh
    # extent 1.6 centred -> to_berkelium_frame keeps the scale; the sliver has float64 area 1.5e-12 (> 1e-12) but
    # its 2e-8 offset is below float32 resolution at 0.5, so in float32 it has zero area.
    v = np.array([[-0.8, -0.8, -0.8], [0.8, -0.8, -0.8], [-0.8, 0.8, 0.8], [0.5, 0.5, 0.5], [0.5 + 2e-8, 0.5, 0.5],
                  [0.5, 0.5, 0.5 + 1.5e-4]])
    f = np.array([[0, 1, 2], [3, 4, 5]])
    assert 0.5 * np.linalg.norm(np.cross(v[4] - v[3], v[5] - v[3])) > 1e-12
    v2, f2, c2, removed = S.finalize_mesh(v, f, None)
    assert removed == 1 and validate_mesh(v2, f2)["degenerate_faces"] == 0 and validate_mesh(v2, f2)["valid"]


# ---------------------------------------------------------------------------------------------- availability
def test_available_reasons(tmp_path):
    ok, why = S.available(tmp_path)
    assert not ok and "weights missing" in why and "text_cond.pt" in why
    z = G.ZImageTripoSR()
    ok, why = z.available()
    assert not ok and why.startswith("GPU-gated")
    with pytest.raises(G.GeneratorUnavailable):
        z.generate("a teapot")
    st = G.generator_status()
    assert set(st) == {G.NATIVE, G.SHAP_E, G.ZIMAGE_TRIPOSR, G.AUTO} and st[G.NATIVE]["available"]
    assert st[G.ZIMAGE_TRIPOSR]["available"] is False


def test_auto_refuses_to_fall_back_when_open_world_is_unavailable(tmp_path):
    auto = G.AutoGenerator(open_world=S.ShapEGenerator(cache_dir=tmp_path))
    with pytest.raises(G.GeneratorUnavailable, match="auto routed to shap-e-text300m"):
        auto.generate("a red mug")
    out = auto.generate("a red sphere")                                  # native route still works
    assert out["report"]["routing"]["selected"] == G.NATIVE and out["vertices"] is not None


# ---------------------------------------------------------------------------- Shap-E with stubbed sampler/decoder
def test_shap_e_generate_with_mocked_weights_records_provenance_and_stages(monkeypatch):
    v, f = box((-0.5, -0.5, -1.0), (0.5, 0.5, 1.0))                       # tall in Shap-E's z
    col = np.random.default_rng(0).random((len(v), 3)).astype(np.float32)
    gen = S.ShapEGenerator(steps=4)
    monkeypatch.setattr(gen, "latent", lambda prompt, seed, cancel, progress: (np.zeros(3), 7))
    monkeypatch.setattr(gen, "decode", lambda lat: (v.astype(np.float64), f, col))
    stages = []
    out = gen.generate("a tall box", seed=3, progress=lambda fr, st: stages.append(st))
    rep, ident = out["report"], gen.identity
    assert stages == ["loading", "decoding", "validating"] and rep["valid"]
    assert rep["generator"] == "shap-e-text300m" and rep["seed"] == 3 and rep["denoiser_calls"] == 7
    assert rep["labels"] == "model-generated, normalized units, not measured geometry"
    assert out["vertices"].max(0)[1] == pytest.approx(0.8)                # tall axis is now +Y, extent 1.6
    assert set(ident["weights_sha256"]) == {"text_cond.pt", "vector_decoder.pt", "ViT-L-14.pt"}
    assert ident["shap_e_commit"] == S.SHAPE_COMMIT and ident["clip_commit"] == S.CLIP_COMMIT
    assert "MIT" in ident["license"] and ident["sampler"]["karras_steps"] == 4
    with pytest.raises(InterruptedError):
        gen.generate("a tall box", cancel=lambda: True)


# --------------------------------------------------------------------------------------- CLIP text-only build
def test_text_only_clip_matches_full_model_bitwise():
    torch = pytest.importorskip("torch")
    clip_model = pytest.importorskip("clip.model")
    import clip
    torch.manual_seed(0)
    full = clip_model.CLIP(embed_dim=32, image_resolution=32, vision_layers=1, vision_width=64, vision_patch_size=16,
                           context_length=77, vocab_size=49408, transformer_width=64, transformer_heads=1,
                           transformer_layers=2).eval()
    text = {k: v for k, v in full.state_dict().items() if k.startswith(S._TEXT_PREFIXES)}
    rebuilt = S._text_only_clip(text, torch.float32)
    tok = clip.tokenize(["a red mug", "a school bus"])
    with torch.no_grad():
        assert torch.equal(full.encode_text(tok), rebuilt.encode_text(tok))
    with pytest.raises(RuntimeError, match="vision tower was dropped"):
        rebuilt.visual(torch.zeros(1, 3, 32, 32))


def test_torchscript_reader_matches_state_dict_and_refuses_foreign_globals(tmp_path):
    torch = pytest.importorskip("torch")
    m = torch.jit.script(torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4)).half())
    p = tmp_path / "m.pt"
    torch.jit.save(m, str(p))
    got, index = S.read_torchscript_tensors(p, lambda n: True)
    ref = m.state_dict()
    assert set(ref) <= set(got) and all(torch.equal(got[k], ref[k]) for k in ref)
    assert index["0.weight"] == ("float16", (4, 3))
    bad = tmp_path / "bad.pt"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("x/data.pkl", pickle.dumps(os.system))
    with pytest.raises(pickle.UnpicklingError, match="refused global"):
        S.read_torchscript_tensors(bad, lambda n: True)


# ------------------------------------------------------------------------------------------- E-T3 evaluation
def test_rotations_24_are_the_distinct_proper_cube_rotations():
    from berkelium_neural.text3d.openworld_eval import rotations_24
    Rs = rotations_24()
    assert len(Rs) == 24 and len({R.tobytes() for R in Rs}) == 24
    assert all(np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1) for R in Rs)


def test_align_and_score_forgives_pose_and_scale_but_not_shape():
    import trimesh

    from berkelium_neural.text3d.openworld_eval import align_and_score
    gt = trimesh.creation.cylinder(radius=0.2, height=1.0, sections=48)
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3]
    pred_v = (np.asarray(gt.vertices) @ R.T) * 3.7 + 5.0           # rotated 90 degrees, scaled, shifted
    same = align_and_score(pred_v, gt.faces, gt.vertices, gt.faces)
    other = trimesh.creation.icosphere(subdivisions=3)
    diff = align_and_score(other.vertices, other.faces, gt.vertices, gt.faces)
    assert same["chamfer_l1"] < 0.01 and same["fscore@0.05"] > 0.99
    assert diff["chamfer_l1"] > 5 * same["chamfer_l1"] and diff["fscore@0.05"] < 0.9
    assert align_and_score(None, None, gt.vertices, gt.faces)["status"] == "no_prediction"


def test_touch_components_merges_touching_pieces_only():
    from berkelium_neural.text3d.openworld_eval import touch_components
    a_v, a_f = box((-1, 0, -0.5), (0, 1, 0.5))
    b_v, b_f = box((0, 0, -0.5), (1, 1, 0.5))                       # shares the x=0 face: touching
    c_v, c_f = box((0.2, 0, -0.5), (1.2, 1, 0.5))                   # 0.2 gap
    touching = (np.concatenate([a_v, b_v]), np.concatenate([a_f, b_f + len(a_v)]))
    apart = (np.concatenate([a_v, c_v]), np.concatenate([a_f, c_f + len(a_v)]))
    assert touch_components(*touching) == 1 and touch_components(*apart) == 2
    assert touch_components(a_v, a_f) == 1 and touch_components(None, None) is None
