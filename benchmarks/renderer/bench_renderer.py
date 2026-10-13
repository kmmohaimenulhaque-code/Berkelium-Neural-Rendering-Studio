"""Renderer benchmark: AMD gsplat vs Berkelium (gsplat forward + Berkelium deterministic HIP backward), with the
Berkelium FP64 reference as the numerical oracle.

Per (scene, resolution) case and per backend:
  correctness  forward RGB/alpha/depth max-abs error vs FP64 oracle; per-parameter-family gradient errors vs FP64
               autograd (spec tolerance |err| <= 2e-5 + 2e-3 |ref|); bitwise gradient repeatability over R runs;
               FP64 finite-difference gradcheck of the oracle itself
  performance  median / p95 forward, backward and end-to-end iteration latency after warmup; peak memory
  workload     Gaussians, visible, Gaussian/tile pairs, max pairs per tile
  quality      fixed-budget reconstruction PSNR on a held-out view (same init, seeds, iterations for all backends)
Backends that are not actually available on this machine are recorded as not_run with the reason; HIP-CPU
timings are recorded as not_applicable (emulation, not GPU performance).
"""

from __future__ import annotations

import hashlib
import resource
import time

import numpy as np
import torch
from berkelium_neural import camera as cam_mod
from berkelium_neural.contracts import RenderConfig
from berkelium_neural.render.interface import GaussianTensors, RendererError
from berkelium_neural.scene import GaussianScene

from benchmarks.common import m, run_metadata, timing_stats

FAMILIES = ("means", "log_scales", "quats", "opacity_logits", "color_logits")
SEMANTICS_NOTE = (
    "All backends use profile gsplat_v1: alpha=min(0.999, o*exp(-sigma)); skip if sigma<0 or alpha<1/255; stop "
    "exclusively when T*(1-alpha)<=1e-4; eps2d=0.3; FOV-clipped Jacobian (1.3x); opacity-aware 3.33-sigma tile "
    "radius; depth-sorted per tile. Known differences: equal-depth ties are broken by Gaussian index in the reference "
    "and by sort stability in gsplat; the reference differentiates the forward with autograd (cumprod), while gsplat "
    "and Berkelium recover transmittance in backward by division (T /= 1-alpha); gsplat accumulates cross-tile "
    "gradients with float atomicAdd, Berkelium with fixed-order per-intersection reductions.")


def synthetic_scene(n: int, seed: int, spread: float = 0.9, z0: float = 3.0) -> GaussianScene:
    rng = np.random.default_rng(seed)
    means = np.c_[rng.uniform(-spread, spread, (n, 2)), z0 + rng.uniform(-0.8, 0.8, n)]
    s = (spread / max(1.0, (n / 200) ** (1 / 3)))
    return GaussianScene(means, np.log(rng.uniform(0.2, 1.0, (n, 3)) * 0.12 * s / 0.9), rng.normal(size=(n, 4)),
                         rng.uniform(-1.5, 3.0, n), rng.normal(size=(n, 3)), meta={"seed": seed, "synthetic": True})


def scene_hash(s: GaussianScene) -> str:
    return hashlib.sha256(s.to_bytes()[0]).hexdigest()


def make_backend(name: str):
    """-> (renderer, device, dtype, timing_meaningful)"""
    from berkelium_neural.render.berkelium_ext import BerkeliumRenderer
    from berkelium_neural.render.gsplat_adapter import GsplatRenderer
    from berkelium_neural.render.reference import ReferenceRenderer
    if name == "reference_fp64":
        return ReferenceRenderer(), "cpu", torch.float64, True
    if name == "reference_cpu":
        return ReferenceRenderer(), "cpu", torch.float32, True
    if name == "berkelium_hipcpu":
        return BerkeliumRenderer("hipcpu"), "cpu", torch.float32, False
    if name == "gsplat":
        return GsplatRenderer("gsplat"), "cuda", torch.float32, True
    if name == "berkelium_rocm":
        return GsplatRenderer("berkelium"), "cuda", torch.float32, True
    raise ValueError(name)


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def _loss_weights(h, w, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(h, w, 3, generator=g), torch.randn(h, w, generator=g), torch.randn(h, w, generator=g) * 0.05


def _forward_backward(r, scene, cam, cfg, dev, dt, W):
    g = GaussianTensors.from_scene(scene, device=dev, dtype=dt, requires_grad=True)
    o = r.render(g, cam, cfg)
    Wc, Wa, Wd = (x.to(dev, dt) for x in W)
    valid = (o.alpha.detach() > 0.5).to(dt)                    # expected depth is only compared where well defined
    loss = (o.rgb * Wc).sum() + (o.alpha * Wa).sum() + (o.depth * Wd * valid).sum()
    loss.backward()
    return o, {f: p.grad.detach().double().cpu() for f, p in zip(FAMILIES, g.params(), strict=True)}


def _depth_row(o, o_ref, valid):
    if not valid.any():
        return m("forward_depth_max_abs_err_alpha>0.5", status="not_applicable", reason="no pixel with alpha>0.5",
                 group="correctness")
    e = (o.depth.double().cpu() - o_ref.depth)[valid].abs()
    tol = 1e-4 + 2e-4 * o_ref.depth[valid].abs()
    return m("forward_depth_max_abs_err_alpha>0.5", float(e.max()), group="correctness",
             tolerance="1e-4+2e-4|ref| (spec valid-depth)", passed=bool((e <= tol).all()))


def grad_errors(got: dict, ref: dict, gate: bool, atol=2e-5, rtol=2e-3) -> list[dict]:
    """Per-family gradient error vs the FP64 oracle. The spec tolerance (|err| <= 2e-5 + 2e-3|ref| per entry) is a
    GATE only on the spec's smooth fixture (gate=True: N<=16, small image); on large scenes the same numbers are
    reported as agreement measurements (passed=None, violations counted) next to the FP32-autograd baseline."""
    out = []
    for f in FAMILIES:
        e = (got[f] - ref[f]).abs()
        tol = atol + rtol * ref[f].abs()
        scale = float(ref[f].abs().max()) + 1e-30
        ok = bool((e <= tol).all())
        out.append(m(f"grad_{f}_max_abs_err", float(e.max()), "", group="correctness", tolerance=f"{atol}+{rtol}|ref|",
                     violations=int((e > tol).sum()), entries=int(e.numel()), ref_max_abs=scale,
                     max_rel_to_ref_max=float(e.max()) / scale, passed=ok if gate else None,
                     gate="spec_smooth_fixture" if gate else "agreement_only"))
    return out


def run_case(case: dict, backends: list[str], proto: dict) -> dict:
    scene = synthetic_scene(case["n"], case["seed"])
    cam = cam_mod.make_camera((0.2, -0.15, -0.6), (0, 0, 3), case["res"], case["res"], fov_y_deg=55)
    cfg = RenderConfig()
    W = _loss_weights(case["res"], case["res"], case["seed"])
    oracle_r, *_ = make_backend("reference_fp64")
    o_ref, g_ref = _forward_backward(oracle_r, scene, cam, cfg, "cpu", torch.float64, W)
    res = {"case": case, "scene_sha256": scene_hash(scene), "backends": {}}
    for b in backends:
        meas = []
        try:
            r, dev, dt, timing_ok = make_backend(b)
            r.prepare(cfg)
        except RendererError as e:
            res["backends"][b] = {"status": "not_run", "reason": f"{e.code}: {e}", "measurements": [
                m(x, status="not_run", reason=e.code, group=gr) for x, gr in (
                    ("forward_rgb_max_abs_err", "correctness"), ("grad_errors", "correctness"),
                    ("gradient_bitwise_repeatable", "correctness"), ("forward_latency", "performance"),
                    ("backward_latency", "performance"), ("iteration_latency", "performance"),
                    ("peak_memory", "performance"))]}
            continue
        try:
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            o, g = _forward_backward(r, scene, cam, cfg, dev, dt, W)
            valid = o_ref.alpha > 0.5
            meas += [m("forward_rgb_max_abs_err", float((o.rgb.double().cpu() - o_ref.rgb).abs().max()), group="correctness",
                       tolerance=2e-4, passed=float((o.rgb.double().cpu() - o_ref.rgb).abs().max()) <= 2e-4),
                     m("forward_alpha_max_abs_err", float((o.alpha.double().cpu() - o_ref.alpha).abs().max()),
                       group="correctness", tolerance=2e-4,
                       passed=float((o.alpha.double().cpu() - o_ref.alpha).abs().max()) <= 2e-4),
                     _depth_row(o, o_ref, valid)]
            meas += grad_errors(g, g_ref, case.get("gate", False))
            hashes = set()
            for _ in range(proto["repeat_determinism"]):
                _, gg = _forward_backward(r, scene, cam, cfg, dev, dt, W)
                hashes.add(hashlib.sha256(b"".join(gg[f].numpy().tobytes() for f in FAMILIES)).hexdigest())
            meas.append(m("gradient_bitwise_repeatable", float(len(hashes) == 1), group="correctness",
                          unique_hashes=len(hashes), repeats=proto["repeat_determinism"]))
            meas += [m("n_gaussians", o.meta.get("n_gaussians"), group="workload"),
                     m("n_visible", o.meta.get("n_visible"), group="workload"),
                     m("n_pairs", o.meta.get("n_pairs"), group="workload"),
                     m("max_pairs_per_tile", o.meta.get("max_pairs_per_tile"), group="workload")]
            if not timing_ok:
                for x in ("forward_latency", "backward_latency", "iteration_latency"):
                    meas.append(m(x, status="not_applicable", reason="HIP-CPU emulation: timing is not GPU performance",
                                  group="performance"))
            else:
                meas += _timings(r, scene, cam, cfg, dev, dt, W, proto)
            if dev == "cuda":
                meas.append(m("peak_memory", torch.cuda.max_memory_allocated() / 2**20, "MiB",
                              method="torch.cuda.max_memory_allocated"))
            else:
                meas.append(m("peak_memory", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "MiB",
                              method="process peak RSS (cumulative, upper bound; not per-backend)"))
            res["backends"][b] = {"status": "measured", "renderer": r.name, "device": dev, "dtype": str(dt),
                                  "measurements": meas}
        except Exception as e:  # noqa: BLE001 - preserved failed run
            res["backends"][b] = {"status": "failed", "reason": f"{type(e).__name__}: {e}", "measurements": meas}
    return res


def _timings(r, scene, cam, cfg, dev, dt, W, proto):
    fw, bw, it = [], [], []
    g = GaussianTensors.from_scene(scene, device=dev, dtype=dt, requires_grad=True)
    opt = torch.optim.Adam(g.params(), lr=1e-3)
    Wc = W[0].to(dev, dt)
    for i in range(proto["warmup"] + proto["repeats"]):
        opt.zero_grad(set_to_none=True)
        _sync(dev)
        t0 = time.perf_counter()
        o = r.render(g, cam, cfg)
        _sync(dev)
        t1 = time.perf_counter()
        (o.rgb * Wc).sum().backward()
        _sync(dev)
        t2 = time.perf_counter()
        opt.step()
        _sync(dev)
        t3 = time.perf_counter()
        if i >= proto["warmup"]:
            fw.append(t1 - t0)
            bw.append(t2 - t1)
            it.append(t3 - t0)
    return [m("forward_latency", timing_stats(fw)["median_ms"], "ms", **timing_stats(fw)),
            m("backward_latency", timing_stats(bw)["median_ms"], "ms", **timing_stats(bw)),
            m("iteration_latency", timing_stats(it)["median_ms"], "ms", **timing_stats(it))]


def fd_gradcheck(seed=0) -> dict:
    from berkelium_neural.render.reference import ReferenceRenderer
    s = synthetic_scene(4, seed, spread=0.25)
    c = cam_mod.make_camera((0, 0, -0.5), (0, 0, 3), 20, 18)
    R, cfg = ReferenceRenderer(), RenderConfig(background=(0.2, 0.1, 0.3))
    Wt = torch.randn(18, 20, 5, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)

    def loss(*p):
        o = R.render(GaussianTensors(*p), c, cfg)
        return (torch.cat([o.rgb, o.alpha[..., None], 0.1 * o.depth[..., None]], -1) * Wt).sum()
    params = [p.detach().clone().requires_grad_(True) for p in GaussianTensors.from_scene(s, dtype=torch.float64)
              .params()]
    ok = torch.autograd.gradcheck(loss, params, eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=False)
    return m("oracle_fd_gradcheck_fp64", float(ok), group="correctness", passed=bool(ok),
             method="torch.autograd.gradcheck central differences eps=1e-6 atol=1e-6 rtol=1e-4, 5 families")


def fit_quality(backend: str, proto: dict, seed: int = 0) -> dict:
    """Fixed-budget reconstruction: fit a perturbed copy of a GT Gaussian scene to K training views; PSNR on a
    held-out view. Same init, views, iterations and optimizer for every backend."""
    from berkelium_neural.text3d.geometry import psnr
    try:
        r, dev, dt, _ = make_backend(backend)
        cfg = RenderConfig(background=(1.0, 1.0, 1.0))
        r.prepare(cfg)
    except RendererError as e:
        return m("heldout_view_psnr_after_fit", status="not_run", reason=e.code, group="quality")
    q = proto["fit"]
    gt = synthetic_scene(q["n"], seed + 100, spread=0.7)
    cams = cam_mod.orbit_cameras(q["views"] + 1, 3.0, 15.0, q["res"], q["res"], target=(0, 0, 3.0))
    train_c, held = cams[:-1], cams[-1]
    with torch.no_grad():
        gtt = GaussianTensors.from_scene(gt, device=dev, dtype=dt)
        targets = [r.render(gtt, c, cfg).rgb for c in train_c]
        held_t = r.render(gtt, held, cfg).rgb
    rng = np.random.default_rng(seed + 200)
    init = GaussianScene(gt.means + rng.normal(0, 0.05, gt.means.shape), gt.log_scales + 0.2, gt.quats,
                         np.zeros(gt.n), rng.normal(size=(gt.n, 3)))
    g = GaussianTensors.from_scene(init, device=dev, dtype=dt, requires_grad=True)
    opt = torch.optim.Adam(g.params(), lr=q["lr"])
    t0 = time.perf_counter()
    for it in range(q["iters"]):
        opt.zero_grad()
        c = train_c[it % len(train_c)]
        (r.render(g, c, cfg).rgb - targets[it % len(train_c)]).abs().mean().backward()
        opt.step()
    with torch.no_grad():
        p = psnr(r.render(g, held, cfg).rgb.cpu().numpy(), held_t.cpu().numpy())
    return m("heldout_view_psnr_after_fit", p, "dB", group="quality", iterations=q["iters"], views=q["views"],
             n=q["n"], res=q["res"], wall_s=round(time.perf_counter() - t0, 2),
             note="GT and fit rendered by the same backend; measures optimisation, not geometric ground truth")


def run(cfg: dict) -> dict:
    out = {"benchmark": "renderer", "semantics": SEMANTICS_NOTE, **run_metadata(cfg, "benchmarks/renderer/"
           "bench_renderer.py"), "correctness_global": [fd_gradcheck()], "cases": [], "quality": {}}
    for case in cfg["cases"]:
        out["cases"].append(run_case(case, cfg["backends"], cfg["protocol"]))
    if cfg["protocol"].get("fit"):
        out["quality"] = {b: fit_quality(b, cfg["protocol"]) for b in cfg["backends"]}
    return out
