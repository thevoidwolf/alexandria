from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ulid import ULID

from alexandria.config import Config
from alexandria.ingest.chunk import chunk_text
from alexandria.ingest.embed import embed_texts
from alexandria.ingest.extract import detect_content_type, extract
from alexandria.ingest.fetch import Fetched, fetch_file, fetch_url
from alexandria.ingest.hash import sha256_bytes, sha256_text


@dataclass
class IngestResult:
    doc_id: str
    was_duplicate: bool
    dedup_gate: str | None  # "source" | "raw" | "text" | None
    title: str | None
    content_type: str
    n_chunks: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _blob_path(cfg: Config, sha: str) -> Path:
    return cfg.blobs_dir / sha[:2] / sha


def _write_blob(cfg: Config, sha: str, data: bytes) -> None:
    if not cfg.storage.keep_blobs:
        return
    p = _blob_path(cfg, sha)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(data)


def _serialize_vec(vec) -> bytes:
    # sqlite-vec accepts raw packed float32 bytes for vec0 columns
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def _record_source(
    conn: sqlite3.Connection, doc_id: str, source_kind: str, source_uri: str
) -> None:
    conn.execute(
        """
        INSERT INTO document_sources(doc_id, source_kind, source_uri, seen_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(doc_id, source_kind, source_uri) DO UPDATE SET seen_at = excluded.seen_at
        """,
        (doc_id, source_kind, source_uri, _now()),
    )
    conn.execute(
        "UPDATE documents SET last_seen_at = ? WHERE id = ?", (_now(), doc_id)
    )


def _add_tags(conn: sqlite3.Connection, doc_id: str, tags: list[str]) -> None:
    for tag in tags:
        conn.execute(
            "INSERT OR IGNORE INTO tags(doc_id, tag) VALUES (?, ?)", (doc_id, tag)
        )


def _lookup_by_source(
    conn: sqlite3.Connection, source_kind: str, source_uri: str
) -> str | None:
    row = conn.execute(
        "SELECT doc_id FROM document_sources WHERE source_kind = ? AND source_uri = ?",
        (source_kind, source_uri),
    ).fetchone()
    return row[0] if row else None


def _lookup_by_raw(conn: sqlite3.Connection, sha: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM documents WHERE sha256_raw = ?", (sha,)
    ).fetchone()
    return row[0] if row else None


def _lookup_by_text(conn: sqlite3.Connection, sha: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM documents WHERE sha256_text = ? LIMIT 1", (sha,)
    ).fetchone()
    return row[0] if row else None


def ingest_fetched(
    fetched: Fetched,
    conn: sqlite3.Connection,
    cfg: Config,
    category: str | None = None,
    tags: list[str] | None = None,
    filename_hint: str | None = None,
) -> IngestResult:
    tags = tags or []

    # Gate 1: source URI already known
    if existing := _lookup_by_source(conn, fetched.source_kind, fetched.source_uri):
        conn.execute("UPDATE documents SET last_seen_at = ? WHERE id = ?", (_now(), existing))
        _add_tags(conn, existing, tags)
        return IngestResult(existing, True, "source", None, "", 0)

    # Gate 2: raw bytes already known
    raw_sha = sha256_bytes(fetched.data)
    if existing := _lookup_by_raw(conn, raw_sha):
        _record_source(conn, existing, fetched.source_kind, fetched.source_uri)
        _add_tags(conn, existing, tags)
        return IngestResult(existing, True, "raw", None, "", 0)

    # Extract
    if filename_hint:
        filename = filename_hint
    elif fetched.source_kind == "file":
        filename = Path(fetched.source_uri).name
    else:
        filename = None
    ctype = detect_content_type(fetched.data, fetched.content_type_hint, filename)
    ex = extract(fetched.data, ctype, cfg)
    if not ex.text.strip():
        if ctype == "pdf":
            raise ValueError(
                f"No extractable text from {fetched.source_uri} "
                "(PDF has no embedded text — likely a scan; enable the marker "
                "backend for OCR: [extractors.pdf] backend = \"marker\")"
            )
        raise ValueError(f"No extractable text from {fetched.source_uri}")

    # Gate 3: normalized text already known
    text_sha = sha256_text(ex.text)
    if existing := _lookup_by_text(conn, text_sha):
        _record_source(conn, existing, fetched.source_kind, fetched.source_uri)
        _add_tags(conn, existing, tags)
        return IngestResult(existing, True, "text", ex.title, ctype, 0)

    # Chunk + embed
    chunks = chunk_text(ex.text, cfg.chunking.tokens, cfg.chunking.overlap)
    embeddings = embed_texts([c.text for c in chunks], cfg.embeddings)

    doc_id = str(ULID())
    now = _now()

    conn.execute("BEGIN")
    try:
        conn.execute(
            """
            INSERT INTO documents(
                id, sha256_raw, sha256_text, content_type, title, author,
                published_at, category, extracted_text, extractor, embed_model,
                bytes, ingested_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id, raw_sha, text_sha, ctype, ex.title, ex.author,
                ex.published_at, category, ex.text, ex.extractor,
                cfg.embeddings.model, len(fetched.data), now, now,
            ),
        )
        conn.execute(
            "INSERT INTO document_sources(doc_id, source_kind, source_uri, seen_at) VALUES (?, ?, ?, ?)",
            (doc_id, fetched.source_kind, fetched.source_uri, now),
        )
        _add_tags(conn, doc_id, tags)

        conn.execute(
            "INSERT INTO documents_fts(doc_id, title, extracted_text) VALUES (?, ?, ?)",
            (doc_id, ex.title or "", ex.text),
        )

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
                (rowid, _serialize_vec(vec)),
            )

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    _write_blob(cfg, raw_sha, fetched.data)

    return IngestResult(
        doc_id=doc_id,
        was_duplicate=False,
        dedup_gate=None,
        title=ex.title,
        content_type=ctype,
        n_chunks=len(chunks),
    )


def ingest_file(
    path: Path,
    conn: sqlite3.Connection,
    cfg: Config,
    category: str | None = None,
    tags: list[str] | None = None,
) -> IngestResult:
    fetched = fetch_file(path)
    return ingest_fetched(fetched, conn, cfg, category=category, tags=tags)


def ingest_url(
    url: str,
    conn: sqlite3.Connection,
    cfg: Config,
    category: str | None = None,
    tags: list[str] | None = None,
) -> IngestResult:
    from alexandria.categorize import categorize_url

    resolved_category, resolved_tags = categorize_url(url, cfg, category, tags)
    fetched = fetch_url(url, cfg.http)
    return ingest_fetched(
        fetched, conn, cfg, category=resolved_category, tags=resolved_tags
    )
