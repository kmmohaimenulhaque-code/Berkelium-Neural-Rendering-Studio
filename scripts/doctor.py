#!/usr/bin/env python3
"""Inspect this machine for Berkelium neural work. Read-only: no installs, no downloads, no GPU allocation beyond
querying device properties (skip even that with --no-gpu-init).

  python scripts/doctor.py            # human summary + verdicts
  python scripts/doctor.py --json     # machine-readable

Verdicts: cpu_reference (CPU tests), hipcpu_kernel (HIP-CPU validation of the Berkelium kernel), rocm_gsplat
(AMD gsplat on ROCm), berkelium_hip_gpu (Berkelium kernel on a real HIP device), cuda_reference (CUDA run; NOT HIP
evidence), studio_e2e (Node app + Playwright).
"""
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMD_GSPLAT_MATRIX = [  # from AMD-Ecosystem/gsplat v1.5.3b2 README
    {"rocm": "7.0", "torch": "2.8", "python": "3.12"}, {"rocm": "6.4", "torch": "2.6", "python": "3.10"}]


def run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return r.stdout.strip() or r.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def mem_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        return None


def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "present")
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-gpu-init", action="store_true")
    a = ap.parse_args(argv)
    r = {"python": platform.python_version(), "os": platform.platform(), "cpus": os.cpu_count(), "ram_gb": mem_gb(),
         "disk_free_gb": round(shutil.disk_usage(ROOT).free / 2**30, 1)}
    r["packages"] = {m: ver(m) for m in ("numpy", "scipy", "skimage", "trimesh", "PIL", "pydantic", "fastapi",
                                         "uvicorn", "httpx", "pytest", "playwright", "torch")}
    t = {}
    if r["packages"]["torch"]:
        import torch
        t = {"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
             "cuda": getattr(torch.version, "cuda", None)}
        if not a.no_gpu_init:
            t["device_visible"] = torch.cuda.is_available()
            if t["device_visible"]:
                t["devices"] = []
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    t["devices"].append({"name": p.name, "total_gb": round(p.total_memory / 2**30, 1),
                                         "arch": getattr(p, "gcnArchName", None)})
    r["torch"] = t
    rocm_ver = None
    for f in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        if Path(f).exists():
            rocm_ver = Path(f).read_text().strip()
            break
    r["rocm"] = {"version": rocm_ver, "rocminfo": bool(shutil.which("rocminfo")), "rocm_smi": bool(shutil.which(
        "rocm-smi")), "hipcc": run(["hipcc", "--version"]) if shutil.which("hipcc") else None,
        "gfx": sorted({w for w in (run(["rocminfo"]) or "").split() if w.startswith("gfx")}) or None}
    r["nvidia_smi"] = bool(shutil.which("nvidia-smi"))
    gs = {"pinned_source": (ROOT / "third_party" / "amd-gsplat" / "gsplat").exists()}
    try:
        import gsplat
        gs.update({"import": True, "version": getattr(gsplat, "__version__", None), "file": gsplat.__file__})
    except Exception as e:  # noqa: BLE001
        gs.update({"import": False, "error": f"{type(e).__name__}: {e}"[:200]})
    r["gsplat"] = gs
    r["hip_cpu"] = {"headers": (ROOT / "third_party" / "hip-cpu" / "include" / "hip" / "hip_runtime.h").exists(),
                    "tbb": any(Path(p).exists() for p in ("/usr/include/tbb/tbb.h", "/usr/include/oneapi/tbb.h")),
                    "cxx": run([os.environ.get("CXX", "g++"), "--version"])}
    r["node"] = {"node": run(["node", "--version"]), "npm": run(["npm", "--version"]),
                 "desktop_node_modules": (ROOT / "apps" / "desktop" / "node_modules").exists()}
    # mixing check: CUDA runtime wheels next to a ROCm torch is an invalid environment
    mixed = None
    if t.get("hip"):
        pl = run([sys.executable, "-m", "pip", "list", "--format=freeze"]) or ""
        nv = [x.split("==")[0] for x in pl.splitlines() if x.lower().startswith(("nvidia-", "cupy-cuda"))]
        mixed = nv or None
    r["mixed_cuda_packages_in_rocm_env"] = mixed
    v = {}
    v["cpu_reference"] = bool(t and r["packages"]["skimage"] and r["packages"]["trimesh"] and r["packages"]["pydantic"])
    v["hipcpu_kernel"] = r["hip_cpu"]["headers"] and r["hip_cpu"]["tbb"] and bool(r["hip_cpu"]["cxx"])
    tv, pv = (t.get("version") or "")[:3], r["python"][:4]
    in_matrix = any(t.get("hip") and tv == m["torch"] and pv.startswith(m["python"]) for m in AMD_GSPLAT_MATRIX)
    v["rocm_gsplat"] = bool(t.get("hip") and t.get("device_visible") and gs.get("import"))
    v["rocm_gsplat_in_amd_support_matrix"] = in_matrix
    v["berkelium_hip_gpu"] = bool(t.get("hip") and t.get("device_visible") and r["rocm"]["hipcc"])
    v["cuda_reference_only"] = bool(t.get("cuda") and t.get("device_visible"))
    v["studio_e2e"] = bool(r["node"]["node"] and r["node"]["desktop_node_modules"] and r["packages"]["playwright"])
    v["environment_valid"] = not mixed
    r["verdicts"] = v
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"python {r['python']} | {r['os']} | {r['cpus']} CPUs | {r['ram_gb']} GB RAM | {r['disk_free_gb']} GB free")
        print(f"torch {t.get('version')} hip={t.get('hip')} cuda={t.get('cuda')} device={t.get('device_visible')} "
              f"{[d['name'] for d in t.get('devices', [])]}")
        print(f"ROCm {r['rocm']['version']} gfx={r['rocm']['gfx']} hipcc={'yes' if r['rocm']['hipcc'] else 'no'} | "
              f"gsplat import={gs.get('import')} {gs.get('version') or gs.get('error', '')}")
        for k, val in v.items():
            print(f"  {'OK ' if val else '-- '} {k}")
        if mixed:
            print(f"  !! CUDA packages present in a ROCm environment: {mixed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
