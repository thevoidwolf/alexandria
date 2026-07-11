"""/api/* endpoint shapes, wired to a real cfg+conn."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from alexandria.ingest import ingest_file
from alexandria.web.api import build_api_routes


@pytest.fixture
def seeded_conn(cfg, conn, corpus_dir: Path, fake_embed):
    ingest_file(corpus_dir / "plain.txt", conn, cfg,
                category="bills", tags=["utility", "electric"])
    ingest_file(corpus_dir / "note.md", conn, cfg,
                category="notes", tags=["design"])
    return conn


@pytest.fixture
def client(cfg, seeded_conn):
    lock = threading.Lock()
    app = Starlette(routes=build_api_routes(cfg, seeded_conn, lock))
    with TestClient(app) as c:
        yield c


def test_info(client, cfg):
    r = client.get("/api/info")
    assert r.status_code == 200
    body = r.json()
    assert body["embed_model"] == cfg.embeddings.model
    assert body["embed_dim"] == cfg.embeddings.dim
    assert body["pdf_backend"] == cfg.extractors.pdf.backend
    assert body["title"] == cfg.web.title
    assert body["max_upload_mb"] == cfg.web.max_upload_mb


def test_catalog(client):
    r = client.get("/api/catalog")
    assert r.status_code == 200
    body = r.json()
    assert body["total_docs"] == 2
    cats = {c["name"]: c["count"] for c in body["categories"]}
    assert cats == {"bills": 1, "notes": 1}


def test_documents_list(client):
    r = client.get("/api/documents?category=bills")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["category"] == "bills"
    assert "utility" in body[0]["tags"]


def test_documents_list_pagination_bounds(client):
    assert client.get("/api/documents?limit=0").status_code == 400
    assert client.get("/api/documents?limit=abc").status_code == 400


def test_document_detail(client):
    doc_id = client.get("/api/documents").json()[0]["id"]
    slim = client.get(f"/api/documents/{doc_id}").json()
    assert slim["extracted_text"] is None
    fat = client.get(f"/api/documents/{doc_id}?include_text=true").json()
    assert fat["extracted_text"]


def test_document_detail_404(client):
    r = client.get("/api/documents/nope")
    assert r.status_code == 404


def test_search_empty_query_returns_empty_list(client):
    r = client.get("/api/search?q=")
    assert r.status_code == 200
    assert r.json() == []


def test_search_rejects_bad_mode(client):
    r = client.get("/api/search?q=foo&mode=bogus")
    assert r.status_code == 400


def test_search_rejects_bad_limit(client):
    assert client.get("/api/search?q=foo&limit=0").status_code == 400
    assert client.get("/api/search?q=foo&limit=9999").status_code == 400


def test_search_returns_hits(client):
    r = client.get("/api/search?q=note&mode=fts")
    assert r.status_code == 200
    hits = r.json()
    assert isinstance(hits, list)
    for h in hits:
        assert {"chunk_id", "doc_id", "score", "snippet", "title",
                "category", "tags", "source_uri", "content_type"} <= h.keys()
