"""Web API + pages for W6 curate slice."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from alexandria.ingest import ingest_file
from alexandria.web.api import build_api_routes
from alexandria.web.auth import COOKIE_NAME, issue_session
from alexandria.web.jobs import JobQueue
from alexandria.web.middleware import SmartAuthMiddleware
from alexandria.web.pages import build_page_routes


@pytest.fixture
def secret() -> bytes:
    return b"z" * 32


@pytest.fixture
def jobs(cfg, conn):
    q = JobQueue(cfg, conn, threading.Lock())
    yield q
    q.stop(timeout=2)


@pytest.fixture
def seeded_client(cfg, conn, jobs, secret, corpus_dir: Path, fake_embed):
    ingest_file(corpus_dir / "plain.txt", conn, cfg,
                category="bills", tags=["utility", "electric"])
    ingest_file(corpus_dir / "note.md", conn, cfg,
                category="notes", tags=["design"])
    lock = threading.Lock()
    routes = build_api_routes(cfg, conn, lock, jobs=jobs) \
        + build_page_routes(cfg, secret, conn, lock, jobs=jobs)
    app = Starlette(routes=routes)
    app.add_middleware(
        SmartAuthMiddleware,
        mcp_token="mcp-t0k",
        session_secret=secret,
        no_auth=False,
    )
    with TestClient(app) as c:
        c.cookies.set(COOKIE_NAME, issue_session(secret, 3600))
        yield c, conn


# ---- PATCH /api/documents/{id} -------------------------------------------


def test_patch_replaces_tags_and_category(seeded_client):
    client, conn = seeded_client
    doc_id = conn.execute("SELECT id FROM documents WHERE category='notes'").fetchone()[0]
    r = client.patch(f"/api/documents/{doc_id}", json={
        "category": "idea", "tags": ["fresh", "draft"],
    })
    assert r.status_code == 200
    row = conn.execute("SELECT category FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert row[0] == "idea"


def test_patch_partial_only_category(seeded_client):
    client, conn = seeded_client
    doc_id = conn.execute("SELECT id FROM documents WHERE category='notes'").fetchone()[0]
    original_tags = sorted(r[0] for r in conn.execute(
        "SELECT tag FROM tags WHERE doc_id=?", (doc_id,)))
    r = client.patch(f"/api/documents/{doc_id}", json={"category": "misc"})
    assert r.status_code == 200
    new_tags = sorted(r[0] for r in conn.execute(
        "SELECT tag FROM tags WHERE doc_id=?", (doc_id,)))
    assert new_tags == original_tags


def test_patch_null_category_unsets(seeded_client):
    client, conn = seeded_client
    doc_id = conn.execute("SELECT id FROM documents WHERE category='notes'").fetchone()[0]
    r = client.patch(f"/api/documents/{doc_id}", json={"category": None})
    assert r.status_code == 200
    row = conn.execute("SELECT category FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert row[0] is None


def test_patch_empty_body_400(seeded_client):
    client, conn = seeded_client
    doc_id = conn.execute("SELECT id FROM documents LIMIT 1").fetchone()[0]
    r = client.patch(f"/api/documents/{doc_id}", json={})
    assert r.status_code == 400


def test_patch_invalid_shape_400(seeded_client):
    client, conn = seeded_client
    doc_id = conn.execute("SELECT id FROM documents LIMIT 1").fetchone()[0]
    r = client.patch(f"/api/documents/{doc_id}", json={"tags": 5})
    assert r.status_code == 400


def test_patch_unknown_404(seeded_client):
    client, _ = seeded_client
    r = client.patch("/api/documents/nope", json={"category": "x"})
    assert r.status_code == 404


# ---- DELETE /api/documents/{id} ------------------------------------------


def test_delete_document(seeded_client):
    client, conn = seeded_client
    doc_id = conn.execute("SELECT id FROM documents WHERE category='notes'").fetchone()[0]
    r = client.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert conn.execute("SELECT 1 FROM documents WHERE id=?", (doc_id,)).fetchone() is None


def test_delete_document_unknown_404(seeded_client):
    client, _ = seeded_client
    r = client.delete("/api/documents/nope")
    assert r.status_code == 404


# ---- taxonomy endpoints --------------------------------------------------


def test_rename_category_endpoint(seeded_client):
    client, conn = seeded_client
    r = client.post("/api/categories/bills/rename", json={"to": "invoices"})
    assert r.status_code == 200
    assert r.json()["affected"] == 1
    row = conn.execute("SELECT category FROM documents WHERE category='invoices'").fetchone()
    assert row is not None


def test_rename_category_blank_400(seeded_client):
    client, _ = seeded_client
    r = client.post("/api/categories/bills/rename", json={"to": "   "})
    assert r.status_code == 400


def test_delete_category_endpoint(seeded_client):
    client, conn = seeded_client
    r = client.delete("/api/categories/bills")
    assert r.status_code == 200
    assert r.json()["affected"] == 1
    remaining = conn.execute("SELECT COUNT(*) FROM documents WHERE category='bills'").fetchone()[0]
    assert remaining == 0


def test_rename_tag_endpoint(seeded_client):
    client, conn = seeded_client
    r = client.post("/api/tags/utility/rename", json={"to": "recurring"})
    assert r.status_code == 200
    assert r.json()["affected"] == 1


def test_delete_tag_endpoint(seeded_client):
    client, conn = seeded_client
    r = client.delete("/api/tags/utility")
    assert r.status_code == 200
    assert r.json()["affected"] == 1


# ---- /taxonomy page + /documents/{id} edit UI ---------------------------


def test_taxonomy_page_renders(seeded_client):
    client, _ = seeded_client
    r = client.get("/taxonomy")
    assert r.status_code == 200
    assert "Categories" in r.text
    assert "Tags" in r.text
    assert 'data-kind="category"' in r.text
    assert 'data-kind="tag"' in r.text
    assert 'href="/taxonomy" class="here"' in r.text


def test_document_detail_shows_edit_and_delete_buttons(seeded_client):
    client, conn = seeded_client
    doc_id = conn.execute("SELECT id FROM documents LIMIT 1").fetchone()[0]
    r = client.get(f"/documents/{doc_id}")
    assert r.status_code == 200
    assert 'id="edit-btn"' in r.text
    assert 'id="delete-btn"' in r.text
    assert 'id="known-categories"' in r.text
