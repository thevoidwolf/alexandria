from __future__ import annotations

import hashlib
import sqlite3
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from stat import S_ISREG

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


class UpdateRefused(Exception):
    """An explicit update (``target_doc_id``) was refused; nothing changed.

    ``doc_id`` names the document the refusal is about.
    """

    def __init__(self, doc_id: str, message: str) -> None:
        super().__init__(message)
        self.doc_id = doc_id


class DocumentNotFound(UpdateRefused):
    def __init__(self, doc_id: str) -> None:
        super().__init__(doc_id, f"document {doc_id} not found")


class ContentConflict(UpdateRefused):
    """The new content is already another document's."""

    def __init__(self, doc_id: str) -> None:
        super().__init__(
            doc_id, f"this content is already stored as document {doc_id}"
        )


class SourceConflict(UpdateRefused):
    """The new file/url source is another document's identity."""

    def __init__(self, doc_id: str) -> None:
        super().__init__(
            doc_id, f"this source already belongs to document {doc_id}"
        )


@dataclass
class IngestResult:
    doc_id: str
    was_duplicate: bool
    dedup_gate: str | None  # "source" | "raw" | "text" | None
    title: str | None
    content_type: str
    n_chunks: int
    was_updated: bool = False  # existing doc's content replaced in place


@dataclass(frozen=True)
class _NewDoc:
    """``_resolve``'s answer when the content becomes a new document.

    ``detach_from`` names a document that also lists the ingested source
    (an edited copy being split off); the new document takes the source
    from it in the same transaction as the insert.
    """

    detach_from: str | None = None


@dataclass(frozen=True)
class _Update:
    """``_resolve``'s answer when ``doc_id`` is updated in place.

    ``keep`` names another ``(source_kind, source_uri)`` of the doc that
    already holds the new bytes (a note and its copy got the same edit);
    it stays attached in its current row so it remains the primary source.
    """

    doc_id: str
    keep: tuple[str, str] | None = None


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


def _detach_source(
    conn: sqlite3.Connection, doc_id: str, source_kind: str, source_uri: str
) -> None:
    conn.execute(
        "DELETE FROM document_sources "
        "WHERE doc_id = ? AND source_kind = ? AND source_uri = ?",
        (doc_id, source_kind, source_uri),
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


def _update_for_edit(
    conn: sqlite3.Connection, doc_id: str, fetched: Fetched, raw_sha: str
) -> _Update | None:
    """How new bytes at a source of ``doc_id`` update it, or None to split off.

    A doc whose content sits at several sources (a note and its copy)
    belongs to its primary source, the oldest one that still exists, so
    editing the copy can't take over the original's id, category and tags.
    A ``file`` whose path is gone doesn't count; other kinds can't be
    checked and always do, as does the source being ingested. If the
    primary file already holds the same new bytes, both got the same edit:
    the doc follows it and the primary keeps its place.
    """
    for kind, uri in conn.execute(
        "SELECT source_kind, source_uri FROM document_sources "
        "WHERE doc_id = ? ORDER BY rowid",
        (doc_id,),
    ).fetchall():
        if (kind, uri) == (fetched.source_kind, fetched.source_uri):
            return _Update(doc_id)
        if kind != "file":
            return None
        path = Path(uri)
        try:
            if not path.exists():
                continue
            # Under the DB lock: only hash a regular file of the right size.
            st = path.stat()
            if not S_ISREG(st.st_mode) or st.st_size != len(fetched.data):
                return None
            with path.open("rb") as f:
                same = hashlib.file_digest(f, "sha256").hexdigest() == raw_sha
        except OSError:
            return None
        return _Update(doc_id, keep=(kind, uri)) if same else None
    return _Update(doc_id)


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
    target_doc_id: str | None = None,
) -> IngestResult | _Update | _NewDoc:
    """Decide what ingesting ``fetched`` does to the store.

    Returns a duplicate ``IngestResult`` (its side effects already applied),
    an ``_Update`` naming the document to update in place, or a ``_NewDoc``
    to insert a new one. Updates and new docs write nothing here, so the answer may
    change between calls; the caller acts only on the final one.
    ``text_sha=None`` (before extraction) skips every text comparison.
    With ``target_doc_id`` the answer is that document or a duplicate of it;
    an ``UpdateRefused`` is raised before anything is written otherwise.
    Call under the DB lock: a text duplicate may swap the doc's blob after
    its own COMMIT.
    """
    kind, uri = fetched.source_kind, fetched.source_uri

    def dup(doc_id: str, gate: str) -> IngestResult:
        return IngestResult(doc_id, True, gate, title, ctype, 0)

    if target_doc_id is not None:
        row = conn.execute(
            "SELECT sha256_raw, sha256_text FROM documents WHERE id = ?",
            (target_doc_id,),
        ).fetchone()
        if row is None:
            raise DocumentNotFound(target_doc_id)
        target_raw, target_text = row
        if (
            kind in IDENTITY_KINDS
            and (owner := _lookup_by_source(conn, kind, uri))
            and owner != target_doc_id
        ):
            raise SourceConflict(owner)
        if raw_sha == target_raw or (text_sha and text_sha == target_text):
            # Nothing to replace: just remember where this copy came from.
            _record_source(conn, target_doc_id, kind, uri)
            _add_tags(conn, target_doc_id, tags)
            return dup(target_doc_id, "raw" if raw_sha == target_raw else "text")
        if (other := _lookup_by_raw(conn, raw_sha)) or (
            text_sha and (other := _lookup_by_text(conn, text_sha))
        ):
            raise ContentConflict(other)
        return _Update(target_doc_id)

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
                _detach_source(conn, owner, kind, uri)
                _record_source(conn, other, kind, uri)
                _add_tags(conn, other, tags)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            return dup(other, gate)
        if update := _update_for_edit(conn, owner, fetched, raw_sha):
            return update
        # An edited copy of the owner's primary source: it becomes its own
        # document and the owner keeps its content, id, category and tags.
        return _NewDoc(detach_from=owner)

    if other := _lookup_by_raw(conn, raw_sha):
        _record_source(conn, other, kind, uri)
        _add_tags(conn, other, tags)
        return dup(other, "raw")
    if text_sha and (other := _lookup_by_text(conn, text_sha)):
        _record_source(conn, other, kind, uri)
        _add_tags(conn, other, tags)
        return dup(other, "text")
    return _NewDoc()


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
    keep: tuple[str, str] | None = None,
) -> str:
    """Replace ``doc_id``'s content with ``fetched``, keeping id, tags and category.

    ``category=None`` leaves the category untouched. The fetched source
    becomes the doc's only source: aliases that pointed at the old bytes no
    longer describe it. ``keep`` names one more source that already holds
    the new bytes; it stays in its row (and so stays primary). Returns the
    doc's previous ``sha256_raw``.
    """
    now = _now()
    conn.execute("BEGIN")
    try:
        row = conn.execute(
            "SELECT sha256_raw FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            # Deleted by another process since _resolve: the lock is per process.
            raise DocumentNotFound(doc_id)
        (old_raw,) = row
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
        if keep:
            conn.execute(
                "DELETE FROM document_sources WHERE doc_id = ? "
                "AND NOT (source_kind = ? AND source_uri = ?)",
                (doc_id, *keep),
            )
        else:
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
    target_doc_id: str | None = None,
) -> IngestResult:
    """Dedup, extract, chunk, embed and store one fetched document.

    New content at a known ``file``/``url`` source updates that document in
    place (same id, tags and category; ``category`` is ignored then), unless
    the source is a copy of the doc's older, still-existing source holding
    different bytes: then it becomes a new document and is detached from
    the old one.

    ``target_doc_id`` names the document ``fetched`` replaces, whatever its
    source; a ``category`` passed then replaces the doc's. Raises
    ``DocumentNotFound`` if it doesn't exist (or is deleted mid-ingest),
    ``SourceConflict`` if a ``file``/``url`` source belongs to another doc
    and ``ContentConflict`` if another doc already holds the content.

    ``lock`` guards ``conn``. It is held only around the DB phases, never
    across extraction or embedding (which can take minutes for a large PDF
    on the marker backend), so other readers aren't starved by a long
    ingest. Because the lock is released mid-ingest, ``_resolve`` re-runs
    just before the write.
    """
    tags = tags or []
    raw_sha = sha256_bytes(fetched.data)

    with lock:
        target = _resolve(
            conn, fetched, raw_sha, None, tags, None, "", cfg, target_doc_id
        )
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
        target = _resolve(
            conn, fetched, raw_sha, text_sha, tags, ex.title, ctype, cfg,
            target_doc_id,
        )
    if isinstance(target, IngestResult):
        return target

    # Chunk + embed (unlocked)
    chunks = chunk_text(ex.text, cfg.chunking.tokens, cfg.chunking.overlap)
    embeddings = embed_texts([c.text for c in chunks], cfg.embeddings)

    doc_id = str(ULID())
    now = _now()

    with lock:
        # Another ingest may have stored or changed this content (or
        # deleted the target) while we were extracting/embedding without
        # the lock.
        target = _resolve(
            conn, fetched, raw_sha, text_sha, tags, ex.title, ctype, cfg,
            target_doc_id,
        )
        if isinstance(target, IngestResult):
            return target
        if isinstance(target, _Update):
            old_raw = _apply_update(
                conn, target.doc_id, fetched, raw_sha, text_sha, ex, ctype,
                chunks, embeddings, tags,
                category=category if target_doc_id else None, cfg=cfg,
                keep=target.keep,
            )
            # Under the lock, after COMMIT, like the insert path below.
            _replace_blob(cfg, old_raw, raw_sha, fetched.data)
            return IngestResult(
                target.doc_id, False, None, ex.title, ctype, len(chunks),
                was_updated=True,
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
            if target.detach_from:
                _detach_source(
                    conn, target.detach_from,
                    fetched.source_kind, fetched.source_uri,
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
    target_doc_id: str | None = None,
) -> IngestResult:
    fetched = fetch_file(path)
    return ingest_fetched(
        fetched, conn, cfg, category=category, tags=tags, lock=lock,
        target_doc_id=target_doc_id,
    )


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
