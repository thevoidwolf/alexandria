"""Shared chunk storage helpers: insert_chunks / delete_chunks."""
from __future__ import annotations

from pathlib import Path

from alexandria.ingest import ingest_file
from alexandria.ingest.store import delete_chunks


def test_delete_chunks_removes_chunks_fts_and_vectors(cfg, conn, corpus_dir: Path, fake_embed):
    doc_id = ingest_file(corpus_dir / "note.md", conn, cfg).doc_id
    assert conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0] > 0

    conn.execute("BEGIN")
    delete_chunks(conn, doc_id)
    conn.execute("COMMIT")

    assert conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()[0] == 1


def test_insert_chunks_links_fts_and_vec_by_rowid(cfg, conn, corpus_dir: Path, fake_embed):
    ingest_file(corpus_dir / "note.md", conn, cfg)

    rowids = [r[0] for r in conn.execute("SELECT rowid FROM chunks")]
    assert rowids
    for rid in rowids:
        assert conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE rowid = ?", (rid,)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM chunks_vec WHERE chunk_rowid = ?", (rid,)
        ).fetchone()[0] == 1
    n = len(rowids)
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == n
    assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == n
