from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    sha256_raw     TEXT NOT NULL UNIQUE,
    sha256_text    TEXT NOT NULL,
    content_type   TEXT NOT NULL,
    title          TEXT,
    author         TEXT,
    published_at   TEXT,
    category       TEXT,
    extracted_text TEXT NOT NULL,
    extractor      TEXT NOT NULL,
    embed_model    TEXT NOT NULL,
    bytes          INTEGER NOT NULL,
    ingested_at    TEXT NOT NULL,
    last_seen_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_sha256_text ON documents(sha256_text);
CREATE INDEX IF NOT EXISTS idx_documents_category    ON documents(category);

CREATE TABLE IF NOT EXISTS document_sources (
    doc_id       TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_kind  TEXT NOT NULL,
    source_uri   TEXT NOT NULL,
    seen_at      TEXT NOT NULL,
    PRIMARY KEY (doc_id, source_kind, source_uri)
);

CREATE INDEX IF NOT EXISTS idx_sources_uri ON document_sources(source_kind, source_uri);

CREATE TABLE IF NOT EXISTS tags (
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag    TEXT NOT NULL,
    PRIMARY KEY (doc_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

CREATE TABLE IF NOT EXISTS chunks (
    id         TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord        INTEGER NOT NULL,
    start_char INTEGER,
    end_char   INTEGER,
    text       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ord);

-- Standalone FTS tables (not external-content). Simpler bookkeeping;
-- rowid is stored explicitly and matches chunks.rowid / documents.rowid
-- via our INSERT ordering. Costs some disk, buys simple code.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    doc_id UNINDEXED,
    title,
    extracted_text,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    text,
    tokenize='porter unicode61'
);
"""


def _create_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            chunk_rowid INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}]
        )
        """
    )


def connect(db_path: Path, embed_dim: int) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")

    conn.executescript(_SCHEMA_SQL)
    _create_vec_table(conn, embed_dim)

    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    elif row[0] != SCHEMA_VERSION:
        raise RuntimeError(
            f"DB schema version {row[0]} does not match code version {SCHEMA_VERSION}"
        )
    return conn
