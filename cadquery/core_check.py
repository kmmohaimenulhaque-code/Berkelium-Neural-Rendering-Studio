"""Berkelium core validation of a sandboxed CadQuery result (the STEP bytes), and IoU against a reference.

The verdict comes only from the pinned core: ``OCCTBackend.measure`` (OCCT volume, bbox, closed shells,
BRepCheck validity, solid count) and ``validation.engine.geometric_results`` (L3: valid + closed + volume > 0;
L4: solid count when an expected count is given), summarised by ``validation.engine.summarize``. Optional extent
requirements (mm, with tolerance) are checked against the core's bounding box.

IoU is exact (OCCT boolean common): ``V(A∩B) / (V(A) + V(B) - V(A∩B))``. ``align="bbox_center"`` first translates
the candidate so its bounding-box centre sits on the reference's (prompts rarely fix absolute placement);
``align="none"`` is the raw-frame IoU. Both are reported by scripts/cq_eval.py; the bbox-centre one is primary."""

from __future__ import annotations

import os
import tempfile


def load_step(step: bytes):
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "in.step")
        with open(p, "wb") as f:
            f.write(step)
        r = STEPControl_Reader()
        if r.ReadFile(p) != IFSelect_ReturnStatus.IFSelect_RetDone:
            raise ValueError("STEP file could not be read")
        r.TransferRoots()
        shape = r.OneShape()
    if shape is None or shape.IsNull():
        raise ValueError("STEP file has no shape")
    return shape


def validate(step: bytes, expected_solids: int | None = None, extents_mm: dict | None = None,
             tol_mm: float = 0.5) -> dict:
    """Core verdict for one result: {"summary", "l3", "results", "measurements"}. Never raises."""
    from berkelium.geometry.occt import OCCTBackend
    from berkelium.schema.common import QuantityModel
    from berkelium.schema.evaluation import Fidelity, ValidationResult
    from berkelium.validation.engine import geometric_results, summarize
    try:
        shape = load_step(step)
        m = OCCTBackend().measure(shape)
    except Exception as e:  # noqa: BLE001 — unreadable STEP is a failure, not a crash
        return {"summary": "fail", "l3": False, "results": [{"level": 3, "status": "fail", "message": str(e)}],
                "measurements": None}
    results = geometric_results("result", m, expected_solids)
    for ax, i in (("x", 0), ("y", 1), ("z", 2)):
        want = (extents_mm or {}).get(ax)
        if want is None:
            continue
        got = m.bbox_max[i] - m.bbox_min[i]
        ok = abs(got - want) <= max(tol_mm, 0.01 * want)
        results.append(ValidationResult(
            validator="requirement.cq@0.1", level=5, status="pass" if ok else "fail", target=f"result.extent_{ax}",
            message=f"extent_{ax} {got:.4g} mm vs {want:.4g} mm", measured=QuantityModel(value=got, unit="mm"),
            fidelity=Fidelity(kind="geometric", method=f"measure@{m.backend}")))
    rep = summarize(results)
    return {"summary": rep.summary, "l3": any(r.level == 3 and r.status == "pass" for r in results),
            "results": [r.model_dump(mode="json") for r in results],
            "measurements": {"backend": m.backend, "volume_mm3": m.volume_mm3, "area_mm2": m.area_mm2,
                             "bbox_min": list(m.bbox_min), "bbox_max": list(m.bbox_max), "n_solids": m.n_solids,
                             "closed": m.closed, "valid": m.valid}}


def _bbox(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box()
    b.SetGap(0.0)
    BRepBndLib.AddOptimal_s(shape, b, False, False)
    lo, hi = b.CornerMin(), b.CornerMax()
    return (lo.X(), lo.Y(), lo.Z()), (hi.X(), hi.Y(), hi.Z())


def _volume(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return max(0.0, p.Mass())


def iou(candidate: bytes, reference: bytes, align: str = "bbox_center") -> float:
    """Exact volumetric IoU of two STEP solids (raises if OCCT cannot intersect them)."""
    from berkelium.geometry.occt import OCCTBackend
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Vec
    a, b = load_step(candidate), load_step(reference)
    if align == "bbox_center":
        (al, ah), (bl, bh) = _bbox(a), _bbox(b)
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(*[(bl[i] + bh[i]) / 2 - (al[i] + ah[i]) / 2 for i in range(3)]))
        a = BRepBuilderAPI_Transform(a, t, True).Shape()
    va, vb = _volume(a), _volume(b)
    if va <= 0 or vb <= 0:
        return 0.0
    inter = OCCTBackend().intersection_volume(a, b)
    union = va + vb - inter
    return float(min(1.0, max(0.0, inter / union))) if union > 0 else 0.0
