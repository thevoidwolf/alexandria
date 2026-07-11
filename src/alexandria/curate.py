"""Curation operations: delete document, rename/delete categories and tags.

All operations are transactional. Renames merge into an existing target when
the target already exists (`INSERT OR IGNORE` for tags, plain UPDATE for
categories since ``documents.category`` isn't unique).

The virtual FTS + vec tables have no foreign keys, so their rows are wiped
explicitly before the base row goes; the ON DELETE CASCADE on
``document_sources``/``tags``/``chunks`` handles the rest.
"""
from __future__ import annotations

import sqlite3

from alexandria.config import Config


def _blob_path(cfg: Config, sha: str):
    return cfg.blobs_dir / sha[:2] / sha


def delete_document(doc_id: str, conn: sqlite3.Connection, cfg: Config) -> bool:
    """Delete a document + all derived rows + its blob. Returns False if unknown."""
    row = conn.execute(
        "SELECT sha256_raw FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        return False
    raw_sha = row[0]

    conn.execute("BEGIN")
    try:
        # sqlite-vec has no doc_id column; scope via chunks.rowid.
        conn.execute(
            "DELETE FROM chunks_vec WHERE chunk_rowid IN "
            "(SELECT rowid FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        )
        conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
        # ON DELETE CASCADE cleans document_sources, tags, chunks.
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    blob = _blob_path(cfg, raw_sha)
    try:
        blob.unlink()
    except FileNotFoundError:
        pass
    return True


def update_document_metadata(
    doc_id: str,
    conn: sqlite3.Connection,
    *,
    category: str | None | type(...) = ...,
    tags: list[str] | None = None,
) -> bool:
    """PATCH semantics: only apply the fields that were passed.

    ``category=...`` (Ellipsis) means "don't touch"; ``category=None`` means
    "unset". ``tags=None`` means "don't touch"; ``tags=[]`` empties the list.
    Returns False if the document doesn't exist.
    """
    exists = conn.execute(
        "SELECT 1 FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    if exists is None:
        return False

    conn.execute("BEGIN")
    try:
        if category is not ...:
            conn.execute(
                "UPDATE documents SET category = ? WHERE id = ?",
                (category, doc_id),
            )
        if tags is not None:
            conn.execute("DELETE FROM tags WHERE doc_id = ?", (doc_id,))
            for tag in tags:
                tag = tag.strip()
                if not tag:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO tags(doc_id, tag) VALUES (?, ?)",
                    (doc_id, tag),
                )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return True


# ---- category ops ---------------------------------------------------------


def rename_category(old: str, new: str, conn: sqlite3.Connection) -> int:
    """Rewrite all ``documents.category = old`` rows to ``new``.

    ``new`` may already be an in-use category; SQLite treats it as a straight
    UPDATE and both original + already-tagged rows continue to co-exist under
    the same label (a natural merge). Returns the number of rows changed.
    """
    if not new.strip():
        raise ValueError("new category name may not be blank")
    cur = conn.execute(
        "UPDATE documents SET category = ? WHERE category = ?",
        (new, old),
    )
    return cur.rowcount


def delete_category(name: str, conn: sqlite3.Connection) -> int:
    """Unset ``documents.category`` on every doc that carries it."""
    cur = conn.execute(
        "UPDATE documents SET category = NULL WHERE category = ?", (name,)
    )
    return cur.rowcount


# ---- tag ops --------------------------------------------------------------


def rename_tag(old: str, new: str, conn: sqlite3.Connection) -> int:
    """Rewrite all ``tags.tag = old`` rows to ``new``.

    If a doc already carries the new tag, the ``PRIMARY KEY(doc_id, tag)``
    constraint would collide; we sidestep by INSERT-OR-IGNORE-then-DELETE.
    Returns the number of docs affected.
    """
    if not new.strip():
        raise ValueError("new tag name may not be blank")
    if old == new:
        return 0
    conn.execute("BEGIN")
    try:
        # Materialize the doc_id set so the two statements see the same rows.
        doc_ids = [r[0] for r in conn.execute(
            "SELECT doc_id FROM tags WHERE tag = ?", (old,)
        )]
        for doc_id in doc_ids:
            conn.execute(
                "INSERT OR IGNORE INTO tags(doc_id, tag) VALUES (?, ?)",
                (doc_id, new),
            )
        conn.execute("DELETE FROM tags WHERE tag = ?", (old,))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(doc_ids)


def delete_tag(name: str, conn: sqlite3.Connection) -> int:
    """Remove a tag from every document that carries it."""
    cur = conn.execute("DELETE FROM tags WHERE tag = ?", (name,))
    return cur.rowcount
