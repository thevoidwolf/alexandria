"""Background ingest jobs for the web UI.

A single worker thread drains the `jobs` SQLite table FIFO, running each job
under the same DB lock the MCP tools use. Progress is reported as coarse
status transitions (queued → running → done|error|cancelled); SSE subscribers
receive one event per transition.

Restart resilience: on `start()`, any `running` row is swept to `error` (the
process died mid-job); `queued` rows are left alone and picked up.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ulid import ULID

from alexandria.config import Config
from alexandria.ingest.fetch import Fetched
from alexandria.ingest.orchestrator import ingest_fetched, ingest_url

# Poll fallback in case a wake event is missed during shutdown races.
_IDLE_POLL_SECONDS = 5.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class JobRow:
    id: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    input: dict
    result: dict | None
    error: str | None


def _row_to_jobrow(row) -> JobRow | None:
    if row is None:
        return None
    return JobRow(
        id=row[0],
        kind=row[1],
        status=row[2],
        created_at=row[3],
        updated_at=row[4],
        input=json.loads(row[5]) if row[5] else {},
        result=json.loads(row[6]) if row[6] else None,
        error=row[7],
    )


class SSEBroadcaster:
    """Cross-thread event fan-out for SSE subscribers.

    The worker runs off the event loop, so it uses ``loop.call_soon_threadsafe``
    to put events on each subscriber's asyncio.Queue.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []

    def subscribe(self, q: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._subs.append((q, loop))

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [(qq, lp) for qq, lp in self._subs if qq is not q]

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q, loop in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                # Loop closed during shutdown; drop.
                pass


class JobQueue:
    def __init__(
        self,
        cfg: Config,
        conn: sqlite3.Connection,
        db_lock: threading.Lock,
    ) -> None:
        self._cfg = cfg
        self._conn = conn
        self._db_lock = db_lock
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._cancel_flags: set[str] = set()
        self._cancel_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self.broadcaster = SSEBroadcaster()

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._sweep_running_to_error()
        self._cfg.pending_uploads_dir.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run, name="alx-jobs", daemon=True
        )
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker:
            self._worker.join(timeout=timeout)

    def _sweep_running_to_error(self) -> None:
        with self._db_lock:
            self._conn.execute(
                "UPDATE jobs SET status='error', error=?, updated_at=? "
                "WHERE status='running'",
                ("server restarted mid-job", _now()),
            )

    # ---- reads -----------------------------------------------------------

    def get(self, job_id: str) -> JobRow | None:
        with self._db_lock:
            row = self._conn.execute(
                "SELECT id, kind, status, created_at, updated_at, "
                "input, result, error FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return _row_to_jobrow(row)

    def list_recent(self, limit: int = 50) -> list[JobRow]:
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT id, kind, status, created_at, updated_at, "
                "input, result, error FROM jobs "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r for r in (_row_to_jobrow(row) for row in rows) if r is not None]

    # ---- enqueue ---------------------------------------------------------

    def enqueue_upload(
        self,
        pending_path: Path,
        filename: str,
        category: str | None,
        tags: list[str],
        content_type_hint: str | None = None,
    ) -> str:
        return self._insert("upload", {
            "pending_path": str(pending_path),
            "filename": filename,
            "category": category,
            "tags": tags,
            "content_type_hint": content_type_hint,
        })

    def enqueue_url(
        self, url: str, category: str | None, tags: list[str]
    ) -> str:
        return self._insert(
            "url", {"url": url, "category": category, "tags": tags}
        )

    def _insert(self, kind: str, input_data: dict) -> str:
        job_id = str(ULID())
        now = _now()
        with self._db_lock:
            self._conn.execute(
                "INSERT INTO jobs(id, kind, status, created_at, updated_at, input) "
                "VALUES (?, ?, 'queued', ?, ?, ?)",
                (job_id, kind, now, now, json.dumps(input_data)),
            )
        self._wake.set()
        self._publish(job_id, "job.queued")
        return job_id

    # ---- cancel ----------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        row = self.get(job_id)
        if row is None or row.status in ("done", "error", "cancelled"):
            return False
        if row.status == "queued":
            self._set_status(job_id, "cancelled")
            self._publish(job_id, "job.cancelled")
            return True
        # 'running' — best-effort cooperative flag; ingest can't be interrupted
        # mid-flight but the worker will honor the flag after this job completes
        # or on the next queued pull if it hasn't started yet.
        with self._cancel_lock:
            self._cancel_flags.add(job_id)
        return True

    def _is_cancelled(self, job_id: str) -> bool:
        with self._cancel_lock:
            return job_id in self._cancel_flags

    def _forget_cancel(self, job_id: str) -> None:
        with self._cancel_lock:
            self._cancel_flags.discard(job_id)

    # ---- worker ----------------------------------------------------------

    def _set_status(
        self,
        job_id: str,
        status: str,
        error: str | None = None,
        result: dict | None = None,
    ) -> None:
        with self._db_lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, error=?, result=?, updated_at=? "
                "WHERE id=?",
                (
                    status,
                    error,
                    json.dumps(result) if result is not None else None,
                    _now(),
                    job_id,
                ),
            )

    def _next_queued(self) -> JobRow | None:
        with self._db_lock:
            row = self._conn.execute(
                "SELECT id, kind, status, created_at, updated_at, "
                "input, result, error FROM jobs "
                "WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
        return _row_to_jobrow(row)

    def _publish(self, job_id: str, event_name: str) -> None:
        row = self.get(job_id)
        if row is None:
            return
        self.broadcaster.publish({"event": event_name, "job": asdict(row)})

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._next_queued()
            if job is None:
                self._wake.wait(timeout=_IDLE_POLL_SECONDS)
                self._wake.clear()
                continue

            if self._is_cancelled(job.id):
                self._forget_cancel(job.id)
                self._set_status(job.id, "cancelled")
                self._publish(job.id, "job.cancelled")
                continue

            self._set_status(job.id, "running")
            self._publish(job.id, "job.running")

            try:
                result = self._process(job)
                self._set_status(job.id, "done", result=result)
                self._publish(job.id, "job.done")
            except Exception as e:
                self._set_status(
                    job.id, "error",
                    error=f"{type(e).__name__}: {e}",
                )
                self._publish(job.id, "job.error")
                # traceback goes to server log for debugging; not stored.
                traceback.print_exc()
            finally:
                self._forget_cancel(job.id)

    # ---- job processors --------------------------------------------------

    def _process(self, job: JobRow) -> dict[str, Any]:
        if job.kind == "upload":
            return self._process_upload(job)
        if job.kind == "url":
            return self._process_url(job)
        raise RuntimeError(f"unknown job kind: {job.kind}")

    def _process_upload(self, job: JobRow) -> dict[str, Any]:
        pending_path = Path(job.input["pending_path"])
        filename = job.input["filename"]
        category = job.input.get("category")
        tags = job.input.get("tags") or []
        content_type_hint = job.input.get("content_type_hint")

        if not pending_path.exists():
            raise FileNotFoundError(
                f"pending upload file gone: {pending_path}"
            )
        data = pending_path.read_bytes()
        fetched = Fetched(
            data=data,
            source_kind="upload",
            source_uri=f"upload:{filename}",
            content_type_hint=content_type_hint,
        )
        with self._db_lock:
            result = ingest_fetched(
                fetched, self._conn, self._cfg,
                category=category, tags=tags,
                filename_hint=filename,
            )
        try:
            pending_path.unlink()
        except FileNotFoundError:
            pass
        return asdict(result)

    def _process_url(self, job: JobRow) -> dict[str, Any]:
        with self._db_lock:
            result = ingest_url(
                job.input["url"], self._conn, self._cfg,
                category=job.input.get("category"),
                tags=job.input.get("tags") or [],
            )
        return asdict(result)
