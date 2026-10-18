#!/usr/bin/env python3
"""Headless demo of the product path, through the same HTTP API the Studio uses.

  bash scripts/start_backend.sh &                         # backend on :8000
  python scripts/demo_text_to_3d.py "a red cone on top of a blue box"
  python scripts/demo_text_to_3d.py "a green capsule next to a white sphere" --fit-steps 100

Submits a text_to_3d job (adapter=auto), prints live progress, then downloads the real artifacts — mesh.glb, the
Gaussian scene and its splat render — into results/demo/<job id>/ and prints their provenance. Nothing is
generated locally: every file comes from the backend's artifact store.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def req(base, path, body=None, raw=False):
    r = urllib.request.Request(f"{base}{path}", data=None if body is None else json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"}, method="POST" if body else "GET")
    with urllib.request.urlopen(r, timeout=60) as resp:
        data = resp.read()
    return data if raw else json.loads(data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--adapter", default="auto")
    ap.add_argument("--fit-steps", type=int, default=0)
    a = ap.parse_args(argv)
    job = req(a.api, "/v1/neural/jobs", {"kind": "text_to_3d", "prompt": a.prompt, "adapter": a.adapter,
                                         "gaussian_fit_steps": a.fit_steps})
    print(f"job {job['id']} submitted")
    last = None
    while True:
        j = req(a.api, f"/v1/neural/jobs/{job['id']}")
        line = f"  {j['state']:<12} {j['stage']:<28} {j['progress'] * 100:5.1f}%"
        if line != last:
            print(line)
            last = line
        if j["state"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.5)
    if j["state"] != "completed":
        print(json.dumps(j.get("error"), indent=2))
        return 1
    out = ROOT / "results" / "demo" / j["id"]
    out.mkdir(parents=True, exist_ok=True)
    for aid in j["artifacts"]:
        rec = req(a.api, f"/v1/neural/artifacts/{aid}")
        for f in rec["files"]:
            if f["name"] in ("mesh.glb", "preview.png", "scene_3dgs.ply", "render.png"):
                (out / f"{rec['kind']}_{f['name']}").write_bytes(
                    req(a.api, f"/v1/neural/artifacts/{aid}/files/{f['name']}", raw=True))
        p = rec["producer"]
        extra = (f"{rec['validation'].get('n_faces')} faces, watertight={rec['validation'].get('watertight')}"
                 if rec["kind"] == "triangle_mesh" else f"{rec['validation'].get('n')} Gaussians"
                 if rec["kind"] == "gaussian_scene" else f"renderer={p.get('renderer')}")
        routed = p.get("adapter_routing", {}).get("reason", "")
        print(f"{rec['kind']:<15} {aid[:16]}  {extra}  {routed}")
    print(f"files: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
