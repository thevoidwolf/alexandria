"""Shared chunk storage helpers: insert_chunks / delete_chunks."""
from __future__ import annotations

from pathlib import Path

from alexandria.ingest import ingest_file
from alexandria.ingest.store import delete_chunks


def _index_rows(conn, doc_id: str) -> tuple[int, int, int]:
    """(chunks, chunks_fts, chunks_vec) row counts for ``doc_id``."""
    return (
        conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0],
        conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0],
        conn.execute(
            "SELECT COUNT(*) FROM chunks_vec WHERE chunk_rowid IN "
            "(SELECT rowid FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        ).fetchone()[0],
    )


def test_delete_chunks_removes_chunks_fts_and_vectors(cfg, conn, corpus_dir: Path, fake_embed):
    doc_id = ingest_file(corpus_dir / "note.md", conn, cfg).doc_id
    other_id = ingest_file(corpus_dir / "plain.txt", conn, cfg).doc_id
    assert _index_rows(conn, doc_id)[0] > 0
    other_before = _index_rows(conn, other_id)
    assert other_before[0] == other_before[1] == other_before[2] > 0

    conn.execute("BEGIN")
    delete_chunks(conn, doc_id)
    conn.execute("COMMIT")

    assert _index_rows(conn, doc_id)[:2] == (0, 0)
    # chunks_vec has no doc_id: only the other doc's vectors may remain.
    assert conn.execute(
        "SELECT COUNT(*) FROM chunks_vec"
    ).fetchone()[0] == other_before[2]
    assert _index_rows(conn, other_id) == other_before
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
