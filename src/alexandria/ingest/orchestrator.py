from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ulid import ULID

from alexandria.config import Config
from alexandria.ingest.chunk import Chunk, chunk_text
from alexandria.ingest.embed import embed_texts
from alexandria.ingest.extract import Extracted, detect_content_type, extract
from alexandria.ingest.fetch import Fetched, fetch_file, fetch_url
from alexandria.ingest.hash import sha256_bytes, sha256_text
from alexandria.ingest.store import delete_chunks, insert_chunks
from alexandria.originals import blob_path

# Source kinds whose URI names one evolving document (a path or URL), so new
# bytes there are an edit. Uploads and MCP texts carry no such identity.
IDENTITY_KINDS = frozenset({"file", "url"})


@dataclass
class IngestResult:
    doc_id: str
    was_duplicate: bool
    dedup_gate: str | None  # "source" | "raw" | "text" | None
    title: str | None
    content_type: str
    n_chunks: int
    was_updated: bool = False  # existing doc's content replaced in place


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_blob(cfg: Config, sha: str, data: bytes) -> None:
    if not cfg.storage.keep_blobs:
        return
    p = blob_path(cfg, sha)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(data)


def _replace_blob(cfg: Config, old_sha: str, new_sha: str, data: bytes) -> None:
    """Swap a doc's blob after its ``sha256_raw`` changed; call after COMMIT."""
    _write_blob(cfg, new_sha, data)
    blob_path(cfg, old_sha).unlink(missing_ok=True)


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


def _refresh_raw(
    conn: sqlite3.Connection,
    doc_id: str,
    old_raw: str,
    fetched: Fetched,
    raw_sha: str,
    tags: list[str],
    cfg: Config,
) -> None:
    """Point ``doc_id`` at new bytes that extract to the text it already has.

    Text, chunks, embeddings and ``ingested_at`` stay as they are.
    """
    conn.execute("BEGIN")
    try:
        conn.execute(
            "UPDATE documents SET sha256_raw = ?, bytes = ?, last_seen_at = ? WHERE id = ?",
            (raw_sha, len(fetched.data), _now(), doc_id),
        )
        _record_source(conn, doc_id, fetched.source_kind, fetched.source_uri)
        _add_tags(conn, doc_id, tags)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    _replace_blob(cfg, old_raw, raw_sha, fetched.data)


def _resolve(
    conn: sqlite3.Connection,
    fetched: Fetched,
    raw_sha: str,
    text_sha: str | None,
    tags: list[str],
    title: str | None,
    ctype: str,
    cfg: Config,
) -> IngestResult | str | None:
    """Decide what ingesting ``fetched`` does to the store.

    Returns a duplicate ``IngestResult`` (its side effects already applied),
    the id of a document to update in place, or ``None`` to insert a new one.
    ``text_sha=None`` (before extraction) skips every text comparison.
    Call under the DB lock: a text duplicate may swap the doc's blob after
    its own COMMIT.
    """
    kind, uri = fetched.source_kind, fetched.source_uri

    def dup(doc_id: str, gate: str) -> IngestResult:
        return IngestResult(doc_id, True, gate, title, ctype, 0)

    if kind in IDENTITY_KINDS and (owner := _lookup_by_source(conn, kind, uri)):
        owner_raw, owner_text = conn.execute(
            "SELECT sha256_raw, sha256_text FROM documents WHERE id = ?", (owner,)
        ).fetchone()
        if raw_sha == owner_raw:
            conn.execute(
                "UPDATE documents SET last_seen_at = ? WHERE id = ?", (_now(), owner)
            )
            _add_tags(conn, owner, tags)
            return dup(owner, "source")
        if text_sha and text_sha == owner_text:
            # Same text, new bytes (CRLF checkout, re-saved PDF). Adopt the
            # bytes so the next scan stops at the source gate instead of
            # extracting again, but only if this source is the owner's one
            # source: text-identical aliases would flip the raw hash back
            # and forth.
            (n_sources,) = conn.execute(
                "SELECT COUNT(*) FROM document_sources WHERE doc_id = ?", (owner,)
            ).fetchone()
            if n_sources == 1 and not _lookup_by_raw(conn, raw_sha):
                _refresh_raw(conn, owner, owner_raw, fetched, raw_sha, tags, cfg)
            else:
                _record_source(conn, owner, kind, uri)
                _add_tags(conn, owner, tags)
            return dup(owner, "text")
        # The source now holds another document's content: move the source
        # there and leave the owner's old content alone.
        gate = "raw"
        other = _lookup_by_raw(conn, raw_sha)
        if not other and text_sha:
            gate, other = "text", _lookup_by_text(conn, text_sha)
        if other:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "DELETE FROM document_sources "
                    "WHERE doc_id = ? AND source_kind = ? AND source_uri = ?",
                    (owner, kind, uri),
                )
                _record_source(conn, other, kind, uri)
                _add_tags(conn, other, tags)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            return dup(other, gate)
        return owner

    if other := _lookup_by_raw(conn, raw_sha):
        _record_source(conn, other, kind, uri)
        _add_tags(conn, other, tags)
        return dup(other, "raw")
    if text_sha and (other := _lookup_by_text(conn, text_sha)):
        _record_source(conn, other, kind, uri)
        _add_tags(conn, other, tags)
        return dup(other, "text")
    return None


def _apply_update(
    conn: sqlite3.Connection,
    doc_id: str,
    fetched: Fetched,
    raw_sha: str,
    text_sha: str,
    ex: Extracted,
    ctype: str,
    chunks: list[Chunk],
    embeddings,
    tags: list[str],
    category: str | None,
    cfg: Config,
) -> str:
    """Replace ``doc_id``'s content with ``fetched``, keeping id, tags and category.

    ``category=None`` leaves the category untouched. The fetched source
    becomes the doc's only source: aliases that pointed at the old bytes no
    longer describe it. Returns the doc's previous ``sha256_raw``.
    """
    now = _now()
    conn.execute("BEGIN")
    try:
        (old_raw,) = conn.execute(
            "SELECT sha256_raw FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        conn.execute(
            """
            UPDATE documents SET
                sha256_raw = ?, sha256_text = ?, content_type = ?, title = ?,
                author = ?, published_at = ?, extracted_text = ?, extractor = ?,
                embed_model = ?, bytes = ?, ingested_at = ?, last_seen_at = ?,
                category = COALESCE(?, category)
            WHERE id = ?
            """,
            (
                raw_sha, text_sha, ctype, ex.title, ex.author, ex.published_at,
                ex.text, ex.extractor, cfg.embeddings.model, len(fetched.data),
                now, now, category, doc_id,
            ),
        )
        delete_chunks(conn, doc_id)
        conn.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
        conn.execute(
            "INSERT INTO documents_fts(doc_id, title, extracted_text) VALUES (?, ?, ?)",
            (doc_id, ex.title or "", ex.text),
        )
        insert_chunks(conn, doc_id, chunks, embeddings)
        conn.execute("DELETE FROM document_sources WHERE doc_id = ?", (doc_id,))
        conn.execute(
            "INSERT INTO document_sources(doc_id, source_kind, source_uri, seen_at) VALUES (?, ?, ?, ?)",
            (doc_id, fetched.source_kind, fetched.source_uri, now),
        )
        _add_tags(conn, doc_id, tags)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return old_raw


def ingest_fetched(
    fetched: Fetched,
    conn: sqlite3.Connection,
    cfg: Config,
    category: str | None = None,
    tags: list[str] | None = None,
    filename_hint: str | None = None,
    lock: AbstractContextManager = nullcontext(),
) -> IngestResult:
    """Dedup, extract, chunk, embed and store one fetched document.

    New content at a known ``file``/``url`` source updates that document in
    place (same id, tags and category; ``category`` is ignored then).

    ``lock`` guards ``conn``. It is held only around the DB phases, never
    across extraction or embedding (which can take minutes for a large PDF
    on the marker backend), so other readers aren't starved by a long
    ingest. Because the lock is released mid-ingest, ``_resolve`` re-runs
    just before the write.
    """
    tags = tags or []
    raw_sha = sha256_bytes(fetched.data)

    with lock:
        target = _resolve(conn, fetched, raw_sha, None, tags, None, "", cfg)
    if isinstance(target, IngestResult):
        return target

    # Extract (unlocked)
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

    text_sha = sha256_text(ex.text)
    with lock:
        target = _resolve(conn, fetched, raw_sha, text_sha, tags, ex.title, ctype, cfg)
    if isinstance(target, IngestResult):
        return target

    # Chunk + embed (unlocked)
    chunks = chunk_text(ex.text, cfg.chunking.tokens, cfg.chunking.overlap)
    embeddings = embed_texts([c.text for c in chunks], cfg.embeddings)

    doc_id = str(ULID())
    now = _now()

    with lock:
        # Another ingest may have stored or changed this content while we
        # were extracting/embedding without the lock.
        target = _resolve(conn, fetched, raw_sha, text_sha, tags, ex.title, ctype, cfg)
        if isinstance(target, IngestResult):
            return target
        if target is not None:
            old_raw = _apply_update(
                conn, target, fetched, raw_sha, text_sha, ex, ctype, chunks,
                embeddings, tags, None, cfg,
            )
            # Under the lock, after COMMIT, like the insert path below.
            _replace_blob(cfg, old_raw, raw_sha, fetched.data)
            return IngestResult(
                target, False, None, ex.title, ctype, len(chunks), was_updated=True
            )

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

            insert_chunks(conn, doc_id, chunks, embeddings)

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Still under the lock so a concurrent delete_document can't unlink
        # the blob between our COMMIT and this write.
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
    lock: AbstractContextManager = nullcontext(),
) -> IngestResult:
    fetched = fetch_file(path)
    return ingest_fetched(fetched, conn, cfg, category=category, tags=tags, lock=lock)


def ingest_url(
    url: str,
    conn: sqlite3.Connection,
    cfg: Config,
    category: str | None = None,
    tags: list[str] | None = None,
    lock: AbstractContextManager = nullcontext(),
) -> IngestResult:
    from alexandria.categorize import categorize_url

    resolved_category, resolved_tags = categorize_url(url, cfg, category, tags)
    fetched = fetch_url(url, cfg.http)
    return ingest_fetched(
        fetched, conn, cfg, category=resolved_category, tags=resolved_tags,
        lock=lock,
    )
