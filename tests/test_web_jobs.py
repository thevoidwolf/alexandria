"""JobQueue behaviour: enqueue, worker processing, restart sweep, cancel, SSE."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from alexandria.web.jobs import JobQueue, SSEBroadcaster, _now


@pytest.fixture
def queue(cfg, conn):
    import threading

    q = JobQueue(cfg, conn, threading.Lock())
    yield q
    q.stop(timeout=2)


# ---- reads / writes without the worker running ----------------------------


def test_enqueue_creates_queued_row(queue):
    job_id = queue.enqueue_url("https://example.com", None, [])
    row = queue.get(job_id)
    assert row is not None
    assert row.status == "queued"
    assert row.kind == "url"
    assert row.input == {"url": "https://example.com", "category": None, "tags": []}


def test_list_recent_orders_newest_first(queue):
    a = queue.enqueue_url("https://a.example", None, [])
    time.sleep(1.01)  # our timestamps are second-precision
    b = queue.enqueue_url("https://b.example", None, [])
    rows = queue.list_recent()
    assert [r.id for r in rows[:2]] == [b, a]


def test_get_unknown_returns_none(queue):
    assert queue.get("nope") is None


def test_cancel_queued_moves_to_cancelled(queue):
    job_id = queue.enqueue_url("https://x.example", None, [])
    assert queue.cancel(job_id) is True
    assert queue.get(job_id).status == "cancelled"
    # Second cancel is a no-op (not cancellable).
    assert queue.cancel(job_id) is False


def test_cancel_unknown_returns_false(queue):
    assert queue.cancel("nope") is False


# ---- restart resilience ---------------------------------------------------


def test_sweep_running_to_error_on_start(queue, conn):
    # Simulate a job that was mid-flight when the process died.
    conn.execute(
        "INSERT INTO jobs(id, kind, status, created_at, updated_at, input) "
        "VALUES ('leftover','url','running',?,?,'{}')", (_now(), _now())
    )
    queue.start()
    row = queue.get("leftover")
    assert row.status == "error"
    assert "restarted" in row.error


# ---- worker: upload end-to-end --------------------------------------------


def _wait_until(pred, timeout: float = 15.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def test_worker_processes_upload_job(cfg, conn, corpus_dir: Path, fake_embed, queue):
    src = corpus_dir / "plain.txt"
    pending = cfg.pending_uploads_dir
    pending.mkdir(parents=True, exist_ok=True)
    pending_file = pending / "test-upload-01"
    pending_file.write_bytes(src.read_bytes())

    queue.start()
    job_id = queue.enqueue_upload(
        pending_path=pending_file,
        filename="plain.txt",
        category="uploads",
        tags=["smoke"],
    )
    assert _wait_until(lambda: queue.get(job_id).status in ("done", "error"))
    row = queue.get(job_id)
    assert row.status == "done", row.error
    assert row.result["was_duplicate"] is False
    assert row.result["content_type"] == "txt"
    # Pending file is cleaned up on success.
    assert not pending_file.exists()


def test_worker_reports_failure_when_pending_file_missing(cfg, conn, fake_embed, queue):
    queue.start()
    ghost = cfg.pending_uploads_dir / "never-existed"
    job_id = queue.enqueue_upload(
        pending_path=ghost, filename="ghost.txt",
        category=None, tags=[],
    )
    assert _wait_until(lambda: queue.get(job_id).status == "error")
    row = queue.get(job_id)
    assert "FileNotFoundError" in row.error


# ---- SSE broadcaster ------------------------------------------------------


def test_broadcaster_delivers_cross_thread():
    b = SSEBroadcaster()

    async def scenario():
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        b.subscribe(q, loop)

        # Publish from another thread (simulating the worker).
        import threading
        threading.Thread(
            target=b.publish, args=({"event": "job.done", "job": {"id": "x"}},),
            daemon=True,
        ).start()

        ev = await asyncio.wait_for(q.get(), timeout=2.0)
        b.unsubscribe(q)
        return ev

    ev = asyncio.run(scenario())
    assert ev == {"event": "job.done", "job": {"id": "x"}}
    assert b.subscriber_count() == 0


def test_broadcaster_survives_dead_loop():
    b = SSEBroadcaster()
    loop = asyncio.new_event_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    b.subscribe(q, loop)
    loop.close()
    # Should not raise.
    b.publish({"event": "x", "job": {}})
