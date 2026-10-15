"""Chunk storage helpers shared by ingest, in-place update and delete.

None of these BEGIN/COMMIT; the caller owns the transaction.
"""
from __future__ import annotations

import sqlite3
import struct

from ulid import ULID

from alexandria.ingest.chunk import Chunk


def serialize_vec(vec) -> bytes:
    # sqlite-vec accepts raw packed float32 bytes for vec0 columns
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def insert_chunks(
    conn: sqlite3.Connection, doc_id: str, chunks: list[Chunk], embeddings
) -> None:
    """Insert chunks plus their FTS and vector rows (linked by chunk rowid)."""
    for c, vec in zip(chunks, embeddings, strict=True):
        chunk_id = str(ULID())
        cur = conn.execute(
            "INSERT INTO chunks(id, doc_id, ord, start_char, end_char, text) VALUES (?, ?, ?, ?, ?, ?)",
            (chunk_id, doc_id, c.ord, c.start_char, c.end_char, c.text),
        )
        rowid = cur.lastrowid
        conn.execute(
            "INSERT INTO chunks_fts(rowid, chunk_id, doc_id, text) VALUES (?, ?, ?, ?)",
            (rowid, chunk_id, doc_id, c.text),
        )
        conn.execute(
            "INSERT INTO chunks_vec(chunk_rowid, embedding) VALUES (?, ?)",
            (rowid, serialize_vec(vec)),
        )


def delete_chunks(conn: sqlite3.Connection, doc_id: str) -> None:
    """Delete a document's chunk, FTS and vector rows."""
    # Scope the index tables via chunks.rowid (so delete chunks last):
    # sqlite-vec has no doc_id column, and chunks_fts.doc_id is UNINDEXED,
    # so filtering on it would scan every chunk in the store.
    conn.execute(
        "DELETE FROM chunks_vec WHERE chunk_rowid IN "
        "(SELECT rowid FROM chunks WHERE doc_id = ?)",
        (doc_id,),
    )
    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN "
        "(SELECT rowid FROM chunks WHERE doc_id = ?)",
        (doc_id,),
    )
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
