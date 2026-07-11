"""Integration test for the get_original MCP tool.

FastMCP's ``@mcp.tool()`` registers the function without wrapping the
callable, so we can drive it directly. State reaches the tool through the
module-level ``_cfg`` / ``_conn`` — we install a seeded conn + isolated cfg
before each test.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from alexandria.ingest import ingest_file
from alexandria import server as srv
from alexandria.web.signed_url import verify as verify_signed_url


@pytest.fixture
def seeded(cfg, conn, corpus_dir: Path, fake_embed, monkeypatch):
    ingest_file(corpus_dir / "plain.txt", conn, cfg, category="notes", tags=["t1"])
    doc_id = conn.execute("SELECT id FROM documents").fetchone()[0]

    # Pin the module-level cfg/conn the tool reads.
    monkeypatch.setattr(srv, "_cfg", cfg, raising=False)
    monkeypatch.setattr(srv, "_conn", conn, raising=False)
    yield cfg, conn, doc_id


def test_get_original_returns_signed_url_when_public_base_set(seeded, monkeypatch):
    cfg, _, doc_id = seeded
    monkeypatch.setattr(
        srv, "_cfg",
        replace(cfg, network=replace(cfg.network,
                                     public_base_url="https://example.tail-net.ts")),
        raising=False,
    )
    r = srv.get_original_tool(doc_id, ttl_seconds=60)
    assert r["found"] is True
    assert r["filename"] == "plain.txt"
    assert r["content_type"].startswith("text/plain")
    assert r["bytes"] > 0
    assert r["url"].startswith(f"https://example.tail-net.ts/files/{doc_id}?exp=")
    assert "&sig=" in r["url"]
    assert r["expires_at"] is not None
    assert r["url_unavailable_reason"] is None
    assert Path(r["path"]).exists()

    # And the URL should actually verify under the same session_secret.
    from urllib.parse import parse_qs, urlsplit
    from alexandria.web.auth import get_or_create_session_secret

    qs = parse_qs(urlsplit(r["url"]).query)
    secret = get_or_create_session_secret(srv._cfg)
    assert verify_signed_url(secret, doc_id, qs["exp"][0], qs["sig"][0])


def test_get_original_falls_back_to_loopback(seeded):
    _, _, doc_id = seeded
    # cfg.network.host defaults to 127.0.0.1; no public_base_url configured.
    r = srv.get_original_tool(doc_id)
    assert r["url"].startswith(f"http://127.0.0.1:8765/files/{doc_id}?exp=")


def test_get_original_reports_missing_base_for_non_loopback(seeded, monkeypatch):
    cfg, _, doc_id = seeded
    monkeypatch.setattr(
        srv, "_cfg",
        replace(cfg, network=replace(cfg.network, host="0.0.0.0")),
        raising=False,
    )
    r = srv.get_original_tool(doc_id)
    assert r["found"] is True
    assert r["url"] is None
    assert "public_base_url" in r["url_unavailable_reason"]
    # local path is still populated so a same-host client can use it.
    assert Path(r["path"]).exists()


def test_get_original_unknown_doc(seeded):
    r = srv.get_original_tool("no-such-doc")
    assert r == {"found": False, "reason": "unknown doc_id"}


def test_get_original_missing_blob(seeded):
    cfg, _, doc_id = seeded
    from alexandria.originals import blob_path, lookup_blob_for
    info = lookup_blob_for(srv._conn, doc_id)
    assert info is not None
    blob_path(cfg, info[0]).unlink()
    r = srv.get_original_tool(doc_id)
    assert r["found"] is False
    assert "missing" in r["reason"]
