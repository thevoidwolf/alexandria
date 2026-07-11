"""W4 pages: /search, /documents, /documents/{id}."""
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
def seeded(cfg, conn, corpus_dir: Path, fake_embed):
    ingest_file(corpus_dir / "plain.txt", conn, cfg,
                category="bills", tags=["utility", "electric"])
    ingest_file(corpus_dir / "second-bill.txt", conn, cfg,
                category="bills", tags=["utility", "water"])
    ingest_file(corpus_dir / "note.md", conn, cfg,
                category="notes", tags=["design"])
    ingest_file(corpus_dir / "paper.md", conn, cfg,
                category="research", tags=["paper"])
    ingest_file(corpus_dir / "page.html", conn, cfg,
                category="web", tags=["reference"])
    return conn


@pytest.fixture
def client(cfg, conn, jobs, secret, seeded):
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
        yield c


# ---- /search ---------------------------------------------------------------


def test_search_empty_renders_form_only(client):
    r = client.get("/search")
    assert r.status_code == 200
    assert 'name="q"' in r.text
    assert "Results" not in r.text
    # nav highlight is present
    assert 'href="/search" class="here"' in r.text


def test_search_with_query_shows_results(client):
    r = client.get("/search?q=bill&mode=fts")
    assert r.status_code == 200
    assert "Results" in r.text
    # At least one link into a document detail page
    assert 'href="/documents/' in r.text


def test_search_filters_persist_in_form(client):
    r = client.get("/search?q=whatever&category=bills&tags=utility&mode=fts")
    assert r.status_code == 200
    assert 'value="whatever"' in r.text
    assert 'value="utility"' in r.text
    # selected="" markers are Jinja-rendered
    assert 'value="bills" selected' in r.text
    assert 'value="fts" selected' in r.text


def test_search_no_hits_shows_empty_message(client):
    # FTS mode so vec KNN doesn't return unrelated neighbours.
    r = client.get("/search?q=zzzzzzzzz-nope&mode=fts")
    assert r.status_code == 200
    assert "No matches" in r.text


# ---- /documents -----------------------------------------------------------


def test_documents_default_lists_all(client):
    r = client.get("/documents")
    assert r.status_code == 200
    assert "5 total" in r.text
    assert 'href="/documents/' in r.text
    assert 'href="/documents" class="here"' in r.text


def test_documents_filter_by_category(client):
    unfiltered = client.get("/documents").text.count('href="/documents/0')
    filtered = client.get("/documents?category=bills")
    assert filtered.status_code == 200
    # Only the two 'bills' docs should link out.
    assert filtered.text.count('href="/documents/0') == 2 < unfiltered
    # The dropdown remembers the selection.
    assert 'value="bills" selected' in filtered.text


def test_documents_pagination_with_21_docs(
    cfg, conn, jobs, secret, corpus_dir: Path, fake_embed, tmp_path: Path
):
    # Write 21 distinct .txt payloads.
    for i in range(21):
        p = tmp_path / f"doc-{i:02d}.txt"
        p.write_text(f"unique payload number {i}\nsome body text\n")
        ingest_file(p, conn, cfg, category="bulk", tags=[])
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
        r = c.get("/documents?category=bulk")
        assert r.status_code == 200
        # Rendered Next link present
        assert 'href="/documents?offset=20' in r.text
        # Next page shows the 21st doc + no further next
        r2 = c.get("/documents?category=bulk&offset=20")
        assert r2.status_code == 200
        assert "Previous" in r2.text


# ---- /documents/{id} -----------------------------------------------------


def _first_doc_id(client) -> str:
    r = client.get("/api/documents?limit=1")
    return r.json()[0]["id"]


def test_document_detail_renders(client):
    doc_id = _first_doc_id(client)
    r = client.get(f"/documents/{doc_id}")
    assert r.status_code == 200
    assert "Sources" in r.text
    assert "Extracted text" in r.text
    # JS fetch target is templated with the actual id.
    assert f"/api/documents/{doc_id}?include_text=true" in r.text


def test_document_detail_404(client):
    r = client.get("/documents/nope-not-real")
    assert r.status_code == 404
    assert "Not found" in r.text
