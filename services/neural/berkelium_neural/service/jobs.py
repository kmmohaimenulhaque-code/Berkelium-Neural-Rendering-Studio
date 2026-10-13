"""Durable job manager: SQLite metadata + transition log, bounded admission, one worker, cancellation,
checkpoint recovery after a restart. Lifecycle states follow decision D-010.

Workers execute only the allowlisted job kinds in EXECUTORS; request fields are never interpreted as commands.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import TERMINAL_STATES, JobRecord, JobState

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL,
  progress REAL NOT NULL, stage TEXT NOT NULL, artifacts TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS transitions (job_id TEXT NOT NULL, at TEXT NOT NULL, from_state TEXT, to_state TEXT NOT NULL,
  detail TEXT);
"""


class Cancelled(Exception):
    pass


class AdmissionError(RuntimeError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JobContext:
    """Handed to executors: progress/stage reporting, cancellation polling, per-job checkpoint directory."""

    def __init__(self, mgr: JobManager, job_id: str, cancel: threading.Event):
        self.mgr, self.job_id, self._cancel = mgr, job_id, cancel
        self.workdir = mgr.root / "jobs" / job_id
        self.workdir.mkdir(parents=True, exist_ok=True)

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self):
        if self._cancel.is_set():
            raise Cancelled()

    def progress(self, frac: float, stage: str):
        self.check()
        self.mgr._update(self.job_id, progress=max(0.0, min(1.0, frac)), stage=stage)

    def checkpointed(self, stage: str, detail: dict | None = None):
        self.mgr._transition(self.job_id, JobState.CHECKPOINTED, detail or {"stage": stage})
        self.mgr._transition(self.job_id, JobState.RUNNING, {"stage": stage})

    def add_artifact(self, aid: str):
        self.mgr._add_artifact(self.job_id, aid)


Executor = Callable[[dict, JobContext], None]


class JobManager:
    def __init__(self, root: Path, executors: dict[str, Executor], max_pending: int = 8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.executors = executors
        self.max_pending = max_pending
        self._db_lock = threading.Lock()
        self._db = sqlite3.connect(self.root / "jobs.sqlite3", check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._cancel: dict[str, threading.Event] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="berkelium-job")
        self._recover()

    # ------------------------------------------------------------------------------------------- db
    def _q(self, sql, args=()):
        with self._db_lock:
            return self._db.execute(sql, args).fetchall()

    def _row(self, r) -> JobRecord:
        return JobRecord(id=r[0], request=json.loads(r[1]), state=JobState(r[2]), progress=r[3], stage=r[4],
                         artifacts=tuple(json.loads(r[5])), error=json.loads(r[6]) if r[6] else None,
                         created_at=r[7], updated_at=r[8])

    def get(self, job_id: str) -> JobRecord | None:
        rows = self._q("SELECT * FROM jobs WHERE id=?", (job_id,))
        return self._row(rows[0]) if rows else None

    def list(self, limit: int = 50) -> list[JobRecord]:
        return [self._row(r) for r in self._q("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))]

    def transitions(self, job_id: str) -> list[dict]:
        return [{"at": a, "from": f, "to": t, "detail": json.loads(d) if d else None}
                for a, f, t, d in self._q("SELECT at, from_state, to_state, detail FROM transitions WHERE job_id=? "
                                          "ORDER BY rowid", (job_id,))]

    def _transition(self, job_id: str, to: JobState, detail: dict | None = None, error: dict | None = None):
        with self._db_lock:
            cur = self._db.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            frm = cur[0] if cur else None
            now = _now()
            self._db.execute("UPDATE jobs SET state=?, updated_at=?, error=COALESCE(?, error) WHERE id=?",
                             (to.value, now, json.dumps(error) if error else None, job_id))
            self._db.execute("INSERT INTO transitions VALUES (?,?,?,?,?)",
                             (job_id, now, frm, to.value, json.dumps(detail) if detail else None))

    def _update(self, job_id: str, **kw):
        sets = ", ".join(f"{k}=?" for k in kw) + ", updated_at=?"
        self._q(f"UPDATE jobs SET {sets} WHERE id=?", (*kw.values(), _now(), job_id))

    def _add_artifact(self, job_id: str, aid: str):
        with self._db_lock:
            arts = json.loads(self._db.execute("SELECT artifacts FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
            if aid not in arts:
                arts.append(aid)
            self._db.execute("UPDATE jobs SET artifacts=?, updated_at=? WHERE id=?", (json.dumps(arts), _now(),
                                                                                     job_id))

    # ------------------------------------------------------------------------------------ lifecycle
    def submit(self, request: dict) -> JobRecord:
        kind = request.get("kind")
        if kind not in self.executors:
            raise AdmissionError("UNSUPPORTED_FEATURE", f"job kind {kind!r} is not enabled")
        pending = self._q("SELECT COUNT(*) FROM jobs WHERE state IN (?,?,?,?)", (JobState.PENDING.value,
                          JobState.RUNNING.value, JobState.CHECKPOINTED.value, JobState.RESUMED.value))[0][0]
        if pending >= self.max_pending:
            raise AdmissionError("RESOURCE_LIMIT", f"{pending} active jobs >= max_pending={self.max_pending}")
        jid = uuid.uuid4().hex
        now = _now()
        self._q("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)", (jid, json.dumps(request), JobState.PENDING.value, 0.0,
                                                                 "queued", "[]", None, now, now))
        self._q("INSERT INTO transitions VALUES (?,?,?,?,?)", (jid, now, None, JobState.PENDING.value, None))
        self._enqueue(jid)
        return self.get(jid)

    def _enqueue(self, jid: str):
        self._cancel[jid] = threading.Event()
        self._pool.submit(self._run, jid)

    def cancel(self, job_id: str) -> JobRecord | None:
        j = self.get(job_id)
        if j is None:
            return None
        if j.state in TERMINAL_STATES:
            return j
        ev = self._cancel.get(job_id)
        if ev:
            ev.set()
        if j.state == JobState.PENDING:
            self._transition(job_id, JobState.CANCELLED, {"by": "user", "before_start": True})
        return self.get(job_id)

    def _run(self, jid: str):
        j = self.get(jid)
        if j is None or j.state in TERMINAL_STATES:
            return
        ev = self._cancel[jid]
        if ev.is_set():
            return
        if j.state != JobState.RESUMED:
            self._transition(jid, JobState.RUNNING)
        else:
            self._transition(jid, JobState.RUNNING, {"resumed": True})
        ctx = JobContext(self, jid, ev)
        try:
            self.executors[j.request["kind"]](j.request, ctx)
            if ev.is_set():
                raise Cancelled()
            self._update(jid, progress=1.0, stage="done")
            self._transition(jid, JobState.COMPLETED)
        except Cancelled:
            self._transition(jid, JobState.CANCELLED, {"by": "user"})
        except Exception as e:  # noqa: BLE001 - recorded as a typed job failure
            code = getattr(e, "code", type(e).__name__)
            self._transition(jid, JobState.FAILED, {"stage": self.get(jid).stage},
                             error={"code": code, "message": str(e)[:2000],
                                    "traceback_tail": traceback.format_exc()[-3000:]})

    def _recover(self):
        """After a restart: jobs that were pending/running/checkpointed are resumed (executors are idempotent and
        reload their own checkpoints from the job workdir); nothing is silently marked completed."""
        rows = self._q("SELECT id, state FROM jobs WHERE state IN (?,?,?,?)", (JobState.PENDING.value,
                       JobState.RUNNING.value, JobState.CHECKPOINTED.value, JobState.RESUMED.value))
        for jid, st in rows:
            self._transition(jid, JobState.RESUMED, {"after_restart": True, "was": st})
            self._enqueue(jid)

    def wait(self, job_id: str, timeout: float = 60.0) -> JobRecord:
        import time
        end = time.time() + timeout
        while time.time() < end:
            j = self.get(job_id)
            if j and j.state in TERMINAL_STATES:
                return j
            time.sleep(0.05)
        return self.get(job_id)

    def shutdown(self, wait: bool = True):
        """Stop accepting work. The running job finishes (wait=True); queued jobs stay PENDING in the database and
        are resumed by the next JobManager on the same root. Nothing is marked cancelled on the user's behalf."""
        self._pool.shutdown(wait=wait, cancel_futures=True)
        with self._db_lock:
            self._db.close()
