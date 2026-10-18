"""Immutable, content-addressed artifact store with provenance records.

Layout under ``root``:
  objects/<sha[:2]>/<sha>     file blobs, named by their own sha256 (write-once)
  records/<id>.json           ArtifactRecord; id = sha256 of the canonical record body (kind, files, parents,
                              producer, config, validation, labels) — identical outputs dedupe, nothing is overwritten
Reads re-verify the blob hash, so a corrupted store is detected instead of served.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import ArtifactFile, ArtifactKind, ArtifactRecord, canonical_json, sha256_hex

_ID = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(RuntimeError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / "records").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _blob(self, sha: str) -> Path:
        return self.root / "objects" / sha[:2] / sha

    def put_blob(self, data: bytes) -> str:
        sha = sha256_hex(data)
        p = self._blob(sha)
        if not p.exists():
            _atomic_write(p, data)
        return sha

    def create(self, kind: ArtifactKind, files: dict[str, tuple[bytes, str]], producer: dict, parents=(),
               config=None, validation=None, labels=None) -> ArtifactRecord:
        afiles = []
        for name in sorted(files):
            data, media = files[name]
            afiles.append(ArtifactFile(name=name, media_type=media, sha256=self.put_blob(data), bytes=len(data)))
        for p in parents:
            if not self.exists(p):
                raise ArtifactError("INVALID_PARENT", f"parent artifact {p} does not exist")
        body = {"kind": ArtifactKind(kind).value, "files": [f.model_dump() for f in afiles], "parents": list(parents),
                "producer": producer, "config": config or {}, "validation": validation or {}, "labels": labels or {}}
        aid = sha256_hex(canonical_json(body))
        with self._lock:
            rp = self.root / "records" / f"{aid}.json"
            if rp.exists():                        # immutable: identical content -> same record
                return self.get(aid)
            rec = ArtifactRecord(id=aid, created_at=_now(), **{**body, "files": tuple(afiles),
                                                                "parents": tuple(parents)})
            _atomic_write(rp, rec.model_dump_json(indent=2).encode())
        return rec

    def exists(self, aid: str) -> bool:
        return bool(_ID.match(aid)) and (self.root / "records" / f"{aid}.json").exists()

    def get(self, aid: str) -> ArtifactRecord:
        if not _ID.match(aid or ""):
            raise ArtifactError("INVALID_ARGUMENT", "artifact id must be 64 lowercase hex characters")
        p = self.root / "records" / f"{aid}.json"
        if not p.exists():
            raise ArtifactError("NOT_FOUND", f"artifact {aid} not found")
        rec = ArtifactRecord.model_validate_json(p.read_text())
        body = {k: v for k, v in json.loads(rec.model_dump_json()).items() if k not in ("id", "created_at",
                                                                                      "schema_version")}
        if sha256_hex(canonical_json(body)) != aid:
            raise ArtifactError("CORRUPTED", f"artifact record {aid} does not match its id")
        return rec

    def read_file(self, aid: str, name: str) -> tuple[bytes, ArtifactFile]:
        rec = self.get(aid)
        for f in rec.files:
            if f.name == name:
                data = self._blob(f.sha256).read_bytes()
                if sha256_hex(data) != f.sha256:
                    raise ArtifactError("CORRUPTED", f"blob for {aid}/{name} failed its sha256 check")
                return data, f
        raise ArtifactError("NOT_FOUND", f"artifact {aid} has no file {name!r}")

    def list(self, limit: int = 50) -> list[ArtifactRecord]:
        ps = sorted((self.root / "records").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        return [self.get(p.stem) for p in ps]
