"""store_document_tool and update reporting on the MCP ingest tools."""
from __future__ import annotations

import dataclasses

import pytest

import alexandria.server as srv
from alexandria.config import IngestConfig, WebConfig
from alexandria.ingest import DocumentNotFound


@pytest.fixture
def use_cfg(monkeypatch, conn):
    def _use(cfg):
        monkeypatch.setattr(srv, "_get", lambda: (cfg, conn))
    return _use


def _sources(conn, doc_id):
    return [tuple(r) for r in conn.execute(
        "SELECT source_kind, source_uri FROM document_sources WHERE doc_id = ?",
        (doc_id,),
    )]


def _text(conn, doc_id):
    return " ".join(r[0] for r in conn.execute(
        "SELECT text FROM chunks WHERE doc_id = ? ORDER BY ord", (doc_id,)
    ))


def test_store_creates_markdown_doc(cfg, conn, use_cfg, fake_embed):
    use_cfg(cfg)
    r = srv.store_document_tool(
        "# Title\n\nbody text", "note.md", category="notes", tags=["a"]
    )
    assert not r["was_duplicate"] and not r["was_updated"]
    assert r["content_type"] == "md"
    assert _sources(conn, r["doc_id"]) == [("mcp", "mcp:note.md")]


def test_store_same_content_is_duplicate(cfg, use_cfg, fake_embed):
    use_cfg(cfg)
    first = srv.store_document_tool("# T\n\nbody", "note.md")
    second = srv.store_document_tool("# T\n\nbody", "note.md")
    assert second["was_duplicate"] and second["dedup_gate"] == "raw"
    assert second["doc_id"] == first["doc_id"]


def test_store_same_filename_new_content_is_new_doc(cfg, use_cfg, fake_embed):
    use_cfg(cfg)
    a = srv.store_document_tool("# T\n\nfirst body", "note.md")
    b = srv.store_document_tool("# T\n\nsecond, different body", "note.md")
    assert a["doc_id"] != b["doc_id"]


def test_store_with_doc_id_updates(cfg, conn, use_cfg, fake_embed):
    use_cfg(cfg)
    a = srv.store_document_tool("# T\n\nfirst body", "note.md")
    b = srv.store_document_tool(
        "# T\n\nbrandnew replacement words", "note.md", doc_id=a["doc_id"]
    )
    assert b["was_updated"] and b["doc_id"] == a["doc_id"]
    assert "brandnew replacement" in _text(conn, a["doc_id"])


@pytest.mark.parametrize("filename", ["x.pdf", "noext", ""])
def test_store_rejects_unsupported_extension(cfg, use_cfg, filename):
    use_cfg(cfg)
    with pytest.raises(ValueError, match="must end in"):
        srv.store_document_tool("hello", filename)


def test_store_rejects_oversize(cfg, use_cfg):
    use_cfg(dataclasses.replace(cfg, web=WebConfig(max_upload_mb=1)))
    with pytest.raises(ValueError, match="exceeds"):
        srv.store_document_tool("a" * (1024 * 1024 + 1), "big.txt")


def test_store_unknown_doc_id_refused(cfg, use_cfg, fake_embed):
    use_cfg(cfg)
    with pytest.raises(DocumentNotFound):
        srv.store_document_tool("hello world", "n.txt", doc_id="nope")


def test_folder_tool_reports_updated(cfg, use_cfg, corpus_dir, fake_embed):
    use_cfg(dataclasses.replace(
        cfg, ingest=IngestConfig(allowed_roots=(str(corpus_dir),))
    ))
    srv.ingest_folder_tool(str(corpus_dir), glob="note.md")
    (corpus_dir / "note.md").write_text("# Note\n\ncompletely rewritten content here\n")
    r = srv.ingest_folder_tool(str(corpus_dir), glob="note.md")
    assert len(r["updated"]) == 1
