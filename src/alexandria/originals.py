"""Helpers for serving the original content-addressed blob of a document.

Shared by the ``/files/<doc_id>`` HTTP route (browser + signed-URL access)
and the ``get_original`` MCP tool. Kept out of ``web/`` so stdio-only MCP
servers don't drag in Starlette just to compute a filename.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlsplit

from alexandria.config import Config

# Explicit map: we control charset and don't rely on mimetypes.guess_type
# (which is unreliable for md and skips charset on POSIX by default).
MIME_BY_CTYPE = {
    "pdf":  ("application/pdf", None),
    "html": ("text/html",       "utf-8"),
    "txt":  ("text/plain",      "utf-8"),
    "md":   ("text/markdown",   "utf-8"),
}


def mime_for(content_type: str) -> str:
    mt, charset = MIME_BY_CTYPE.get(content_type, ("application/octet-stream", None))
    return f"{mt}; charset={charset}" if charset else mt


def filename_from_source(source_uri: str | None, content_type: str, doc_id: str) -> str:
    """Best-effort human-friendly filename for the download.

    Basename-only so it can't smuggle path traversal into a
    Content-Disposition header. Falls back to ``<doc_id>.<ext>``.
    """
    ext = content_type if content_type in MIME_BY_CTYPE else "bin"
    fallback = f"{doc_id}.{ext}"
    if not source_uri:
        return fallback

    if source_uri.startswith("upload:"):
        candidate = source_uri.split(":", 1)[1]
    elif source_uri.startswith(("http://", "https://")):
        parts = urlsplit(source_uri)
        candidate = unquote(parts.path.rsplit("/", 1)[-1]) if parts.path else ""
    else:
        candidate = Path(source_uri).name

    candidate = (candidate or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if not candidate or candidate in {".", ".."}:
        return fallback
    return candidate


def blob_path(cfg: Config, sha: str) -> Path:
    return cfg.blobs_dir / sha[:2] / sha


def lookup_blob_for(
    conn: sqlite3.Connection, doc_id: str
) -> tuple[str, str, str | None] | None:
    """Return (sha256_raw, content_type, first_source_uri) or None if unknown."""
    row = conn.execute(
        "SELECT sha256_raw, content_type FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        return None
    src = conn.execute(
        "SELECT source_uri FROM document_sources WHERE doc_id = ? "
        "ORDER BY seen_at LIMIT 1",
        (doc_id,),
    ).fetchone()
    return row[0], row[1], (src[0] if src else None)
