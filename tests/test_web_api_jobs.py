"""/api/upload, /api/ingest-url, /api/jobs* wired to a real JobQueue."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from alexandria.web.api import build_api_routes
from alexandria.web.jobs import JobQueue


@pytest.fixture
def jobs(cfg, conn):
    q = JobQueue(cfg, conn, threading.Lock())
    yield q
    q.stop(timeout=2)


@pytest.fixture
def client(cfg, conn, jobs, fake_embed):
    lock = threading.Lock()  # separate from queue's lock is fine for this test
    app = Starlette(routes=build_api_routes(cfg, conn, lock, jobs=jobs))
    with TestClient(app) as c:
        yield c


def _wait_status(client, job_id: str, *, terminal=("done", "error", "cancelled"),
                 timeout: float = 15.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        if r.status_code == 200 and r.json()["status"] in terminal:
            return r.json()["status"]
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} didn't reach terminal state")


def test_info_shows_jobs_enabled(client):
    assert client.get("/api/info").json()["jobs_enabled"] is True


def test_upload_single_file(client, jobs, corpus_dir: Path):
    jobs.start()
    plain = corpus_dir / "plain.txt"
    r = client.post(
        "/api/upload",
        files={"files": ("plain.txt", plain.read_bytes(), "text/plain")},
        data={"category": "uploads", "tags": "smoke,test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["errors"] == []
    assert len(body["job_ids"]) == 1
    status = _wait_status(client, body["job_ids"][0])
    assert status == "done"


def test_upload_multiple_files_one_job_each(client, jobs, corpus_dir: Path):
    jobs.start()
    r = client.post(
        "/api/upload",
        files=[
            ("files", ("plain.txt", (corpus_dir / "plain.txt").read_bytes(), "text/plain")),
            ("files", ("note.md", (corpus_dir / "note.md").read_bytes(), "text/markdown")),
        ],
        data={"category": "uploads"},
    )
    assert r.status_code == 200
    ids = r.json()["job_ids"]
    assert len(ids) == 2
    for job_id in ids:
        assert _wait_status(client, job_id) == "done"


def test_upload_requires_files_field(client):
    r = client.post("/api/upload", data={"category": "x"})
    assert r.status_code == 400


def test_upload_enforces_size_cap_via_streaming(cfg, conn, jobs, fake_embed):
    """Payload under the 2× fast-reject threshold but over the streaming cap."""
    from dataclasses import replace
    tiny_cfg = replace(cfg, web=replace(cfg.web, max_upload_mb=1))
    # 1.5 MB — under the 2× fast-reject threshold, over the 1 MB stream cap.
    big = b"x" * (int(1.5 * 1024 * 1024))

    app = Starlette(routes=build_api_routes(
        tiny_cfg, conn, threading.Lock(), jobs=jobs,
    ))
    with TestClient(app) as c:
        r = c.post(
            "/api/upload",
            files={"files": ("big.txt", big, "text/plain")},
            data={},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["job_ids"] == []
        assert body["errors"][0]["reason"].endswith("MB cap")


def test_upload_fast_rejects_oversize_content_length(cfg, conn, jobs, fake_embed):
    """Content-Length > 2× cap short-circuits to 413 before reading."""
    from dataclasses import replace
    tiny_cfg = replace(cfg, web=replace(cfg.web, max_upload_mb=1))
    huge = b"x" * (3 * 1024 * 1024)

    app = Starlette(routes=build_api_routes(
        tiny_cfg, conn, threading.Lock(), jobs=jobs,
    ))
    with TestClient(app) as c:
        r = c.post(
            "/api/upload",
            files={"files": ("huge.txt", huge, "text/plain")},
        )
        assert r.status_code == 413


def test_ingest_url_enqueues(client, jobs):
    r = client.post(
        "/api/ingest-url",
        json={"url": "https://example.com/whatever", "category": "web", "tags": ["a", "b"]},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    row = jobs.get(job_id)
    assert row.kind == "url"
    assert row.input["url"] == "https://example.com/whatever"
    assert row.input["tags"] == ["a", "b"]


def test_ingest_url_validates_scheme(client):
    r = client.post("/api/ingest-url", json={"url": "ftp://x"})
    assert r.status_code == 400
    r = client.post("/api/ingest-url", json={})
    assert r.status_code == 400


def test_list_jobs_returns_recent(client, jobs):
    a = jobs.enqueue_url("https://a.example", None, [])
    b = jobs.enqueue_url("https://b.example", None, [])
    r = client.get("/api/jobs?limit=5")
    assert r.status_code == 200
    ids = [row["id"] for row in r.json()]
    assert set(ids) >= {a, b}


def test_get_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_cancel_queued_job(client, jobs):
    job_id = jobs.enqueue_url("https://never.example", None, [])
    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200
    assert jobs.get(job_id).status == "cancelled"
    # Second cancel returns 409.
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409
