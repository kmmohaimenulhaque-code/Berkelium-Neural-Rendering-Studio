#!/usr/bin/env python3
"""Federated GPU execution on Kaggle: this host plans, packages and verifies; a Kaggle GPU kernel computes.

  python scripts/kaggle_gpu_runner.py check                           # credentials, CLI, account access (no GPU)
  python scripts/kaggle_gpu_runner.py run   configs/kaggle/<job>.json  # bundle -> push -> poll -> pull -> verify
  python scripts/kaggle_gpu_runner.py status configs/kaggle/<job>.json
  python scripts/kaggle_gpu_runner.py pull  configs/kaggle/<job>.json [--admit]   # resume after a local restart

A job spec (configs/kaggle/*.json) names the code paths to ship, pip pins, ordered steps (a step may run several
commands in parallel, one per GPU) and the output globs to bring back. The runner:

1. Bundles the working tree's files under ``include`` (tracked + untracked-not-ignored, so uncommitted work ships too)
   into a byte-reproducible tar.gz and records its sha256. The bundle is embedded in the kernel script, which
   verifies the hash before extracting: the remote never runs code other than what this host hashed.
2. Pushes one Kaggle script kernel (GPU + internet) and polls its state with exponential backoff (30 s -> 5 min).
3. On completion downloads the kernel output: ``receipt.json`` (environment, per-step rc/time/log tail),
   ``manifest.json`` (sha256 + size of every output file, computed remotely) and the files. Every file is re-hashed
   here; any mismatch, missing or unlisted file fails the pull. Nothing is admitted into the repository unless the
   whole manifest verifies (``--admit``), and an admitted file never overwrites a different existing file.
4. Fails fast: a failed remote step stops the remaining steps, the receipt carries the log tail, and the local exit
   code is non-zero. There is no automatic retry.

State lives in results/kaggle/<job>/state.json (kernel ref, version, bundle sha256, spec sha256), so ``run`` is
idempotent: an identical job (same spec + bundle) that already completed is not pushed again unless ``--force``.
Kaggle exposes logs only after a run ends, so the full remote log is saved then (remote.log), not streamed live.

Credentials: KAGGLE_USERNAME plus KAGGLE_KEY (or KAGGLE_API_TOKEN / KAGGLE_TOKEN). They are passed to the Kaggle CLI
through its environment and never written to disk, receipts or logs.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = ROOT / "results" / "kaggle"
MAX_BUNDLE_BYTES = 4 * 2**20          # the bundle is embedded in the kernel source; keep the code-only ship small
ACCELERATORS = {"t4x2": "NvidiaTeslaT4", "p100": "NvidiaTeslaP100"}


class RunnerError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- credentials / CLI
def kaggle_env() -> dict:
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_TOKEN")
    if not user or not key:
        raise RunnerError("KAGGLE_USERNAME and KAGGLE_KEY (or KAGGLE_API_TOKEN / KAGGLE_TOKEN) must be set")
    env = dict(os.environ)
    env.update({"KAGGLE_USERNAME": user, "KAGGLE_KEY": key, "KAGGLE_API_TOKEN": key})
    return env


def kaggle(*args, timeout=600, check=True) -> str:
    exe = shutil.which("kaggle")
    if not exe:
        raise RunnerError("kaggle CLI not found: pip install -r configs/environments/kaggle-control.txt")
    r = subprocess.run([exe, *args], env=kaggle_env(), capture_output=True, text=True, timeout=timeout, check=False)
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        raise RunnerError(f"kaggle {' '.join(args[:2])} failed (rc={r.returncode}): {out[-800:]}")
    return out


def account_handle() -> str:
    """The URL handle that owns kernels (kernel refs are <handle>/<slug>). KAGGLE_USERNAME is sometimes set to the
    display name; the token alone identifies the account, so the handle is read back from the API (or KAGGLE_OWNER)."""
    if os.environ.get("KAGGLE_OWNER"):
        return os.environ["KAGGLE_OWNER"]
    out = kaggle("kernels", "list", "--mine", "--page-size", "5", "--csv")
    refs = [ln.split(",")[0] for ln in out.splitlines()[1:] if "/" in ln.split(",")[0]]
    owners = {r.split("/")[0] for r in refs}
    if len(owners) != 1:
        raise RunnerError(f"cannot determine the account handle from the kernel listing ({owners or 'no kernels'}); "
                          "set KAGGLE_OWNER")
    return owners.pop()


def cmd_check(_a) -> int:
    env = kaggle_env()
    handle = account_handle()
    print(json.dumps({"handle": handle, "cli": kaggle("--version"), "account_access": "ok",
                      "note": None if env["KAGGLE_USERNAME"] == handle else
                      f"KAGGLE_USERNAME is {env['KAGGLE_USERNAME']!r}, not the handle; the token authenticates and "
                      "the handle is read from the API"}, indent=2))
    return 0


# --------------------------------------------------------------------------------------------------------- bundle
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_files(spec: dict) -> list[str]:
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard", "--", *spec["include"]],
                       capture_output=True, text=True, check=True)
    extra, missing = set(), []
    for pat in spec.get("include_files", []):         # exact paths must exist; glob patterns may match nothing
        hits = [str(q.relative_to(ROOT)) for q in ROOT.glob(pat) if q.is_file()]
        if not hits and not any(c in pat for c in "*?["):
            missing.append(pat)
        extra.update(hits)
    if missing:
        raise RunnerError(f"include_files missing locally: {missing}")
    files = sorted({f for f in r.stdout.splitlines() if (ROOT / f).is_file()} | extra)
    return files


def make_bundle(spec: dict) -> tuple[bytes, list[str]]:
    """Deterministic tar.gz (sorted names, zeroed mtimes/owners, gzip mtime 0): same tree -> same sha256."""
    files = bundle_files(spec)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for f in files:
            data = (ROOT / f).read_bytes()
            ti = tarfile.TarInfo(f)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
            if os.access(ROOT / f, os.X_OK):
                ti.mode = 0o755
            tar.addfile(ti, io.BytesIO(data))
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0, compresslevel=9) as g:
        g.write(raw.getvalue())
    blob = gz.getvalue()
    if len(blob) > MAX_BUNDLE_BYTES:
        raise RunnerError(f"bundle is {len(blob) / 2**20:.1f} MiB (> {MAX_BUNDLE_BYTES / 2**20:.0f} MiB): ship code "
                          "and "
                          "configs only; large inputs are fetched remotely by pinned URL + sha256")
    return blob, files


def git_state() -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len(g("status", "--porcelain").splitlines())}


# ------------------------------------------------------------------------------------------------- remote script
REMOTE = r'''
# Generated by scripts/kaggle_gpu_runner.py. Verifies the embedded code bundle, runs the job steps in order (fail
# fast), and writes receipt.json + manifest.json (sha256 of every output file) to /kaggle/working/out.
import base64, glob, hashlib, io, json, os, platform, shutil, subprocess, sys, tarfile, time
SPEC = json.loads(base64.b64decode(SPEC_B64))
BUNDLE = base64.b64decode(BUNDLE_B64)
OUT, WORK = "/kaggle/working/out", SPEC.get("workdir", "/tmp/berkelium")
os.makedirs(OUT + "/logs", exist_ok=True)
receipt = {"job": SPEC["name"], "run_id": RUN_ID, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "bundle_sha256_expected": BUNDLE_SHA256, "steps": [], "status": "running"}
def save():
    open(OUT + "/receipt.json", "w").write(json.dumps(receipt, indent=2, default=str))
def sh(c, **kw):
    return subprocess.run(c, shell=True, capture_output=True, text=True, **kw).stdout.strip()
got = hashlib.sha256(BUNDLE).hexdigest()
receipt["bundle_sha256"] = got
if got != BUNDLE_SHA256:
    receipt["status"] = "failed"; receipt["error"] = "bundle sha256 mismatch"; save(); raise SystemExit(0)
shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)
tarfile.open(fileobj=io.BytesIO(BUNDLE), mode="r:gz").extractall(WORK, filter="data")
env = {"python": platform.python_version(), "host": platform.node(),
       "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
       "torch": sh(sys.executable + " -c 'import torch;"
                   "print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'"),
       "disk_free_gb": {p: round(shutil.disk_usage(p).free / 1e9, 1) for p in ("/kaggle/working", "/tmp")},
       "mem_total_mib": int(open("/proc/meminfo").readline().split()[1]) // 1024, "cpu_count": os.cpu_count()}
receipt["environment"] = env; save()
if SPEC.get("data"):          # large inputs shipped as a private Kaggle dataset: verify each file, then place it
    src = glob.glob("/kaggle/input/*/data_manifest.json") + glob.glob("/kaggle/input/**/data_manifest.json",
                                                                      recursive=True)
    man = json.load(open(src[0])) if src else None
    if man is None or man["sha256"] != DATA_SHA256:
        receipt["status"] = "failed"; receipt["error"] = "data manifest missing or not the one pushed"; save()
        raise SystemExit(0)
    base = os.path.dirname(src[0]); bad = []
    with tarfile.open(os.path.join(base, "data.bin")) as tf:
        tf.extractall(WORK, filter="data")
    for rel, h in man["files"].items():
        if hashlib.sha256(open(os.path.join(WORK, rel), "rb").read()).hexdigest() != h:
            bad.append(rel)
    receipt["data"] = {"files": len(man["files"]), "bad": bad}; save()
    if bad:
        receipt["status"] = "failed"; receipt["error"] = f"{len(bad)} data files fail their sha256"; save()
        raise SystemExit(0)
def run_step(i, step):
    cmds = step["parallel"] if "parallel" in step else [step["cmd"]]
    t0 = time.time(); procs = []
    for j, c in enumerate(cmds):
        log = f"{OUT}/logs/{i:02d}_{step['name']}" + (f"_{j}" if len(cmds) > 1 else "") + ".log"
        fh = open(log, "w")
        procs.append((c, log, fh, subprocess.Popen(c, shell=True, cwd=WORK, stdout=fh, stderr=subprocess.STDOUT,
                                                   env={**os.environ, "PYTHONUNBUFFERED": "1", **step.get("env", {})})))
    res = []
    for c, log, fh, p in procs:
        rc = p.wait(); fh.close()
        tail = open(log, errors="replace").read()[-3000:]
        res.append({"cmd": c, "rc": rc, "log": os.path.relpath(log, OUT), "tail": tail})
        print(f"[{step['name']}] rc={rc}: {c}\n{tail[-1500:]}", flush=True)
    ok = all(r["rc"] == 0 or step.get("allow_fail") for r in res)
    receipt["steps"].append({"name": step["name"], "seconds": round(time.time() - t0, 1), "ok": ok, "commands": res})
    save(); return ok
receipt["status"] = "completed"
for i, step in enumerate(SPEC["steps"]):
    if not run_step(i, step):
        receipt["status"] = "failed"; receipt["failed_step"] = step["name"]; break
files = []
for pat in SPEC["outputs"]:
    for f in sorted(glob.glob(os.path.join(WORK, pat), recursive=True)):
        if os.path.isfile(f):
            rel = os.path.relpath(f, WORK); dst = os.path.join(OUT, "files", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(f, dst); files.append(rel)
manifest = {}
for rel in sorted(set(files)):
    b = open(os.path.join(OUT, "files", rel), "rb").read()
    manifest[rel] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
open(OUT + "/manifest.json", "w").write(json.dumps(manifest, indent=2))
receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
receipt["outputs"] = len(manifest); save()
print("BERKELIUM_JOB_STATUS", receipt["status"], flush=True)
'''


def upload_data(spec: dict, run_dir: Path) -> tuple[str | None, str | None]:
    """spec["data"] globs -> data.tar + data_manifest.json (sha256 per file) uploaded as a private Kaggle dataset
    <handle>/berkelium-data-<name> (a new version per run). -> (dataset ref, manifest sha256)."""
    if not spec.get("data"):
        return None, None
    files = sorted({str(q.relative_to(ROOT)) for pat in spec["data"] for q in ROOT.glob(pat) if q.is_file()})
    if not files:
        raise RunnerError(f"spec data globs match nothing: {spec['data']}")
    d = run_dir / "dataset"
    d.mkdir(parents=True, exist_ok=True)
    with tarfile.open(d / "data.bin", "w") as tar:
        for f in files:
            tar.add(ROOT / f, arcname=f)
    man = {"files": {f: sha256_file(ROOT / f) for f in files}}
    man["sha256"] = sha256_bytes(json.dumps(man["files"], sort_keys=True).encode())
    (d / "data_manifest.json").write_text(json.dumps(man, indent=1))
    ref = f"{account_handle()}/berkelium-data-{spec['name']}"
    (d / "dataset-metadata.json").write_text(json.dumps({"title": f"berkelium-data-{spec['name']}", "id": ref,
                                                         "licenses": [{"name": "other"}]}))
    exists = ref in kaggle("datasets", "list", "--mine", "--csv", "-s", f"berkelium-data-{spec['name']}", check=False)
    if exists:
        kaggle("datasets", "version", "-p", str(d), "-m", run_dir.name, timeout=3600)
    else:
        kaggle("datasets", "create", "-p", str(d), timeout=3600)
    for _ in range(60):                       # wait until Kaggle has processed the upload
        st = kaggle("datasets", "status", ref, check=False)
        if "ready" in st.lower():
            break
        time.sleep(20)
    else:
        raise RunnerError(f"dataset {ref} not ready: {st}")
    print(f"uploaded {len(files)} data files as {ref} (manifest {man['sha256'][:16]}…)")
    return ref, man["sha256"]


def build_kernel(spec: dict, blob: bytes, run_id: str, kdir: Path, slug: str, data_ref=None, data_sha=None):
    kdir.mkdir(parents=True, exist_ok=True)
    head = (f"SPEC_B64 = {base64.b64encode(json.dumps(spec).encode()).decode()!r}\n"
            f"BUNDLE_SHA256 = {sha256_bytes(blob)!r}\nRUN_ID = {run_id!r}\nDATA_SHA256 = {data_sha!r}\n"
            f"BUNDLE_B64 = {base64.b64encode(blob).decode()!r}\n")
    (kdir / "run.py").write_text(head + REMOTE)
    title = spec.get("title", slug) if spec.get("slug", slug) == slug else slug    # Kaggle: title must give the slug
    meta = {"id": f"{account_handle()}/{slug}", "title": title, "code_file": "run.py",
            "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [data_ref] if data_ref else [], "competition_sources": [], "kernel_sources": []}
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return meta["id"]


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


# --------------------------------------------------------------------------------------------------------- state
def load_spec(path: str) -> tuple[dict, str]:
    p = Path(path)
    spec = json.loads(p.read_text())
    for k in ("name", "include", "steps", "outputs"):
        if k not in spec:
            raise RunnerError(f"{p}: job spec needs {k!r}")
    return spec, sha256_bytes(p.read_bytes())


def state_dir(spec) -> Path:
    d = STATE_ROOT / spec["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_state(spec) -> dict:
    p = state_dir(spec) / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def write_state(spec, st: dict):
    (state_dir(spec) / "state.json").write_text(json.dumps(st, indent=2))


def kernel_status(ref: str) -> str:
    out = kaggle("kernels", "status", ref, timeout=120)
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in out.upper():
            return s.lower()
    return "unknown:" + out[-200:]


def poll(ref: str, max_hours: float) -> str:
    delay, t_end = 30, time.time() + max_hours * 3600
    while True:
        s = kernel_status(ref)
        print(f"{time.strftime('%H:%M:%S')} {ref}: {s}", flush=True)
        if s in ("complete", "error", "cancel"):
            return s
        if time.time() > t_end:
            raise RunnerError(f"{ref} still {s} after {max_hours} h of polling (the remote run continues; "
                              "resume with `status`/`pull`)")
        time.sleep(delay)
        delay = min(300, int(delay * 1.6))


# ---------------------------------------------------------------------------------------------- pull and verify
def pull(spec: dict, st: dict, admit: bool, partial: bool = False) -> dict:
    run_dir = state_dir(spec) / st["run_id"]
    stage = run_dir / "remote"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    kaggle("kernels", "output", st["kernel"], "-p", str(stage), timeout=3600)
    for lg in stage.glob("*.log"):                     # Kaggle's own run log (JSON stream) -> plain text
        try:
            text = "".join(e.get("data", "") for e in json.loads(lg.read_text()))
        except ValueError:
            text = lg.read_text(errors="replace")
        (run_dir / "remote.log").write_text(text)
    out = stage / "out"
    if not (out / "receipt.json").exists():
        raise RunnerError(f"no receipt in kernel output (see {run_dir / 'remote.log'})")
    receipt = json.loads((out / "receipt.json").read_text())
    if receipt.get("run_id") != st["run_id"] or receipt.get("bundle_sha256") != st["bundle_sha256"]:
        # Kaggle keeps running an earlier version when a new one is pushed to a busy kernel; its output must never be
        # admitted under this run's id (found 2026-10-16: an fp16 run's NaN output pulled for a bf16 push).
        (run_dir / "receipt.REJECTED_stale_run.json").write_text(json.dumps(receipt, indent=2, default=str))
        raise RunnerError(f"kernel output is from run {receipt.get('run_id')} (bundle "
                          f"{str(receipt.get('bundle_sha256'))[:16]}), not this push ({st['run_id']}, "
                          f"{st['bundle_sha256'][:16]}): nothing admitted")
    manifest = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    bad, present = [], set()
    for rel, m in manifest.items():
        f = out / "files" / rel
        present.add(rel)
        if not f.is_file():
            bad.append(f"missing {rel}")
        elif f.stat().st_size != m["bytes"] or sha256_file(f) != m["sha256"]:
            bad.append(f"checksum mismatch {rel}")
    extra = [str(p.relative_to(out / "files")) for p in (out / "files").rglob("*") if p.is_file()
             and str(p.relative_to(out / "files")) not in present] if (out / "files").exists() else []
    bad += [f"unlisted file {x}" for x in extra]
    verify = {"files": len(manifest), "verified": len(manifest) - len(bad), "problems": bad,
              "bytes": sum(m["bytes"] for m in manifest.values())}
    (run_dir / "receipt.json").write_text(json.dumps({"local": {**st, "verification": verify}, "remote": receipt,
                                                      "manifest": manifest}, indent=2, default=str))
    if receipt.get("status") != "completed" and not (partial and receipt.get("failed_step")):
        failed = receipt.get("failed_step") or receipt.get("error")
        tail = next((c["tail"] for s in receipt.get("steps", []) if not s["ok"] for c in s["commands"]
                     if c["rc"] != 0), "")
        raise RunnerError(f"remote job {receipt.get('status')} at {failed!r}; log tail:\n{tail[-2000:]}")
    if bad:
        raise RunnerError(f"output verification failed: {bad[:10]}")
    admitted, conflicts = [], []
    if admit:
        for rel, m in manifest.items():
            dst = ROOT / rel
            if dst.exists() and sha256_file(dst) != m["sha256"]:
                conflicts.append(rel)
        if conflicts:
            raise RunnerError(f"refusing to admit: {len(conflicts)} files differ from existing ones, "
                              f"e.g. {conflicts[:5]}")
        for rel in manifest:
            dst = ROOT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(out / "files" / rel, dst)
                admitted.append(rel)
    st.update({"status": "pulled" if receipt.get("status") == "completed" else
               f"pulled_partial (failed at {receipt.get('failed_step')})",
               "verified_files": verify["verified"], "admitted": len(admitted)})
    write_state(spec, st)
    print(json.dumps({"job": spec["name"], "remote_status": receipt["status"], "verification": verify,
                      "admitted": len(admitted), "receipt": _rel(run_dir / "receipt.json")},
                     indent=2))
    return receipt


# ------------------------------------------------------------------------------------------------------ commands
def cmd_run(a) -> int:
    spec, spec_sha = load_spec(a.spec)
    blob, files = make_bundle(spec)
    bsha = sha256_bytes(blob)
    st = read_state(spec)
    if st.get("bundle_sha256") == bsha and st.get("spec_sha256") == spec_sha and st.get("status") == "pulled" \
            and not a.force:
        print(f"{spec['name']}: identical job already completed and pulled ({st['run_id']}); --force to rerun")
        return 0
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    slug = a.slug or spec.get("slug") or f"berkelium-{spec['name']}"
    try:                                      # a push to a busy kernel does not start; refuse instead of mixing runs
        busy = kernel_status(f"{account_handle()}/{slug}") in ("running", "queued")
    except RunnerError:
        busy = False
    if busy:
        raise RunnerError(f"kernel {slug} is still running an earlier version; wait for it or use --slug")
    kdir = state_dir(spec) / run_id / "kernel"
    data_ref, data_sha = upload_data(spec, state_dir(spec) / run_id)
    ref = build_kernel(spec, blob, run_id, kdir, slug, data_ref, data_sha)
    acc = ACCELERATORS[a.accelerator or spec.get("accelerator", "t4x2")]
    st = {"job": spec["name"], "run_id": run_id, "kernel": ref, "accelerator": acc, "spec": a.spec,
          "spec_sha256": spec_sha, "bundle_sha256": bsha, "bundle_bytes": len(blob), "bundle_files": len(files),
          "data": {"dataset": data_ref, "manifest_sha256": data_sha} if data_ref else None,
          "git": git_state(), "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "pushed"}
    (state_dir(spec) / run_id / "bundle_files.txt").write_text("\n".join(files) + "\n")
    out = kaggle("kernels", "push", "-p", str(kdir), "--accelerator", acc, timeout=600)
    st["push_response"] = out[-300:]
    write_state(spec, st)
    print(f"pushed {ref} ({len(files)} files, bundle {len(blob) / 1024:.0f} KiB sha256 {bsha[:16]}…, {acc})")
    if a.no_wait:
        return 0
    s = poll(ref, a.max_hours)
    st["kernel_final_status"] = s
    write_state(spec, st)
    pull(spec, st, a.admit)
    return 0


def cmd_status(a) -> int:
    spec, _ = load_spec(a.spec)
    st = read_state(spec)
    if not st:
        raise RunnerError(f"{spec['name']}: never pushed")
    print(json.dumps({**{k: st[k] for k in ("job", "run_id", "kernel", "status") if k in st},
                      "kernel_status": kernel_status(st["kernel"])}, indent=2))
    return 0


def cmd_pull(a) -> int:
    spec, _ = load_spec(a.spec)
    st = read_state(spec)
    if not st:
        raise RunnerError(f"{spec['name']}: never pushed")
    s = kernel_status(st["kernel"])
    if s not in ("complete", "error", "cancel"):
        raise RunnerError(f"{st['kernel']} is {s}; pull after it ends")
    pull(spec, st, a.admit, getattr(a, "admit_partial", False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("spec")
    r.add_argument("--slug", help="Kaggle kernel slug (default: the spec's slug or berkelium-<name>)")
    r.add_argument("--accelerator", choices=sorted(ACCELERATORS))
    r.add_argument("--admit", action="store_true", help="copy verified outputs into the repository")
    r.add_argument("--force", action="store_true")
    r.add_argument("--no-wait", action="store_true", help="push only; use status/pull later")
    r.add_argument("--max-hours", type=float, default=11.5, help="local polling limit")
    for name in ("status", "pull"):
        p = sub.add_parser(name)
        p.add_argument("spec")
        if name == "pull":
            p.add_argument("--admit", action="store_true")
            p.add_argument("--admit-partial", action="store_true",
                           help="also admit the verified outputs of a run whose LAST steps failed (recorded)")
    a = ap.parse_args(argv)
    try:
        return {"check": cmd_check, "run": cmd_run, "status": cmd_status, "pull": cmd_pull}[a.cmd](a)
    except RunnerError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
