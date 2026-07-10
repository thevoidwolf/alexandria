"""MCP stdio server exposing the six Alexandria tools.

Concurrency: FastMCP runs tools on anyio threads, so a single sqlite3 connection
is opened with `check_same_thread=False` and guarded by a lock. WAL journaling
lets multiple readers coexist; writes serialize behind the lock. This is
appropriate for a single-user local server.
"""
from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from alexandria.catalog import get_catalog, get_document, list_documents
from alexandria.config import Config, load as load_config
from alexandria.db import connect
from alexandria.ingest import ingest_folder, ingest_url
from alexandria.search import search

_cfg: Config | None = None
_conn = None
_lock = threading.Lock()


def _get():
    global _cfg, _conn
    if _cfg is None:
        _cfg = load_config()
    if _conn is None:
        _conn = connect(_cfg.db_path, _cfg.embeddings.dim)
    return _cfg, _conn


mcp = FastMCP("alexandria")


@mcp.tool()
def ingest_folder_tool(
    path: str,
    recursive: bool = True,
    category: str | None = None,
    tags: list[str] | None = None,
    glob: str | None = None,
) -> dict[str, Any]:
    """Walk a folder and ingest all supported files (pdf, html, txt, md).

    Args:
        path: Absolute or ~-expanded folder path.
        recursive: Descend into subdirectories.
        category: Optional category to apply to every new document.
        tags: Optional tags to apply to every new document.
        glob: Optional filename glob (e.g. "*.pdf"). Overrides the default extension filter.

    Returns:
        {"ingested": [doc_id...], "duplicates": [doc_id...], "errors": [{path, reason}], "scanned": N}
    """
    cfg, conn = _get()
    with _lock:
        result = ingest_folder(
            Path(path), conn, cfg,
            recursive=recursive, glob=glob,
            category=category, tags=tags or [],
        )
    return {
        "ingested": result.ingested,
        "duplicates": result.duplicates,
        "errors": [{"path": p, "reason": r} for p, r in result.errors],
        "scanned": result.scanned,
    }


@mcp.tool()
def ingest_url_tool(
    url: str,
    category: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch a URL and ingest its content (HTML, PDF, text, or markdown).

    If category is omitted, URL-based heuristics from config.toml may supply one.

    Returns:
        {"doc_id", "was_duplicate", "dedup_gate", "title", "content_type", "n_chunks"}
    """
    cfg, conn = _get()
    with _lock:
        r = ingest_url(url, conn, cfg, category=category, tags=tags or [])
    return {
        "doc_id": r.doc_id,
        "was_duplicate": r.was_duplicate,
        "dedup_gate": r.dedup_gate,
        "title": r.title,
        "content_type": r.content_type,
        "n_chunks": r.n_chunks,
    }


@mcp.tool()
def search_tool(
    query: str,
    category: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
    mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """Search the knowledge store.

    Args:
        query: Free-text query.
        category: Restrict to a single category.
        tags: Restrict to documents carrying all given tags (AND).
        limit: Max hits to return.
        mode: "hybrid" (default), "fts" (keyword only), or "vec" (semantic only).

    Returns a list of hits, each with chunk_id, doc_id, score, snippet, title,
    category, tags, source_uri, and content_type.
    """
    cfg, conn = _get()
    with _lock:
        hits = search(
            query, conn, cfg,
            category=category, tags=tags or [],
            limit=limit, mode=mode,  # type: ignore[arg-type]
        )
    return [
        {
            "chunk_id": h.chunk_id,
            "doc_id": h.doc_id,
            "score": h.score,
            "snippet": h.snippet,
            "title": h.title,
            "category": h.category,
            "tags": h.tags,
            "source_uri": h.source_uri,
            "content_type": h.content_type,
        }
        for h in hits
    ]


@mcp.tool()
def get_document_tool(
    doc_id: str, include_text: bool = False
) -> dict[str, Any] | None:
    """Fetch a document's full metadata by id.

    Set include_text=True to also return the normalized extracted text. Returns
    None if the document is not found.
    """
    _, conn = _get()
    with _lock:
        doc = get_document(doc_id, conn, include_text=include_text)
    return asdict(doc) if doc else None


@mcp.tool()
def list_documents_tool(
    category: str | None = None,
    tags: list[str] | None = None,
    since: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List document metadata (no body text), most-recently-ingested first.

    Args:
        category: Filter to a single category.
        tags: AND-filter — docs must have all given tags.
        since: ISO8601 timestamp; only docs ingested at or after.
        limit / offset: Pagination.
    """
    _, conn = _get()
    with _lock:
        docs = list_documents(
            conn, category=category, tags=tags or [],
            since=since, limit=limit, offset=offset,
        )
    return [asdict(d) for d in docs]


@mcp.tool()
def get_catalog_tool() -> dict[str, Any]:
    """Return facet counts and totals for the whole corpus.

    Returns:
        {"total_docs", "total_bytes", "categories": [{name, count}], "tags": [{name, count}]}
    """
    _, conn = _get()
    with _lock:
        summary = get_catalog(conn)
    return asdict(summary)


def run() -> None:
    """Entry point: run the MCP stdio server."""
    mcp.run()


if __name__ == "__main__":
    run()
