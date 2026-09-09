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

from datetime import datetime, timezone

from alexandria import anchors as _anchors
from alexandria.catalog import get_catalog, get_document, list_documents
from alexandria.classify import Suggestion, suggest_metadata, suggest_metadata_bulk
from alexandria.config import Config, load as load_config
from alexandria.curate import (
    delete_category as _delete_category,
    delete_document as _delete_document,
    delete_tag as _delete_tag,
    rename_category as _rename_category,
    rename_tag as _rename_tag,
    update_document_metadata as _update_document_metadata,
)
from alexandria.db import connect
from alexandria.ingest import ingest_folder, ingest_url
from alexandria.originals import (
    blob_path,
    filename_from_source,
    lookup_blob_for,
    mime_for,
)
from alexandria.search import format_snippet_markdown, search

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

    Returns a list of hits, each with chunk_id, doc_id, score, fts_rank,
    vec_rank, matched_in, snippet, title, display_title, category, tags,
    source_uri, and content_type.

    Confidence signal: RRF flattens ``score`` into a narrow band
    (~1/(60+rank)), so use ``fts_rank`` / ``vec_rank`` (1-indexed within
    each retriever's over-fetch window; None if not in that window) and
    ``matched_in`` (e.g. ``["fts", "vec"]``) to gauge how strong a hit is.
    A chunk in ``matched_in=["fts", "vec"]`` with both ranks near 1 is
    much stronger than one with only a single low rank.

    Title fields: ``title`` is the raw extractor metadata (may be null).
    ``display_title`` is always populated — prefer it for user-facing
    rendering; it falls back to the source URI's basename (filename or
    URL path segment) and then to ``"untitled <content_type>"``.
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
            "fts_rank": h.fts_rank,
            "vec_rank": h.vec_rank,
            "matched_in": list(h.matched_in),
            "snippet": format_snippet_markdown(h.snippet),
            "title": h.title,
            "display_title": h.display_title,
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


@mcp.tool()
def delete_document_tool(doc_id: str) -> dict[str, Any]:
    """Permanently delete a document, all its chunks, FTS/vec index rows, and blob.

    Returns {"deleted": True} on success or {"deleted": False, "reason": ...} if
    the document is unknown. Destructive: no undo.
    """
    cfg, conn = _get()
    with _lock:
        ok = _delete_document(doc_id, conn, cfg)
    if not ok:
        return {"deleted": False, "reason": "unknown doc_id"}
    return {"deleted": True}


_LOOPBACK_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}


def _resolve_public_base_url(cfg: Config) -> str | None:
    """Full URL prefix (no trailing slash) to prepend to /files/... links.

    Order:
    1. ``[network] public_base_url`` if set — the honest answer for
       Tailscale / LAN / reverse-proxy deploys.
    2. ``http://<host>:<port>`` only when host is loopback — fine for dev.
    3. None otherwise, since we can't guess a routable name for 0.0.0.0
       or a non-loopback bind. Caller should surface this to the user.
    """
    if cfg.network.public_base_url:
        return cfg.network.public_base_url.rstrip("/")
    if cfg.network.host in _LOOPBACK_HOSTNAMES:
        return f"http://{cfg.network.host}:{cfg.network.port}"
    return None


@mcp.tool()
def get_original_tool(
    doc_id: str, ttl_seconds: int = 3600
) -> dict[str, Any]:
    """Get a fetchable link (and local path) for a document's original file.

    Useful when the extracted text is ambiguous and the caller needs to
    consult the source directly. Blobs are content-addressed by sha256 and
    kept by default (``storage.keep_blobs = true``).

    Args:
        doc_id: The document id (as returned by ``search``/``list_documents``).
        ttl_seconds: How long the signed URL should stay valid. Clamped to
            [1, 86400]. Defaults to 3600 (1 hour).

    Returns:
        ``{"found": True, "filename", "content_type", "bytes", "url",
        "expires_at", "path", "url_unavailable_reason"}``
        or ``{"found": False, "reason": ...}`` if the document/blob is gone.

        ``url`` is null when ``[network] public_base_url`` isn't set and the
        server isn't on loopback — set that config knob to enable remote
        fetching. ``path`` is a local filesystem path, useful only for MCP
        clients on the same host as the server.
    """
    from alexandria.web.auth import get_or_create_session_secret
    from alexandria.web.signed_url import clamp_ttl, sign

    cfg, conn = _get()
    with _lock:
        info = lookup_blob_for(conn, doc_id)
    if info is None:
        return {"found": False, "reason": "unknown doc_id"}

    sha, ctype, src_uri = info
    if not cfg.storage.keep_blobs:
        return {"found": False, "reason": "original not retained (storage.keep_blobs = false)"}
    blob = blob_path(cfg, sha)
    if not blob.exists():
        return {"found": False, "reason": "original blob missing on disk"}

    filename = filename_from_source(src_uri, ctype, doc_id)
    size = blob.stat().st_size

    base = _resolve_public_base_url(cfg)
    url: str | None = None
    expires_at: str | None = None
    url_unavailable_reason: str | None = None

    if not cfg.web.enabled:
        url_unavailable_reason = (
            "web routes disabled ([web] enabled = false); use the local path"
        )
    elif base is None:
        url_unavailable_reason = (
            "no public URL: set [network] public_base_url to your reachable "
            "hostname (e.g. https://alexandria.example.com)"
        )
    else:
        secret = get_or_create_session_secret(cfg)
        exp, sig = sign(secret, doc_id, ttl_seconds=clamp_ttl(ttl_seconds))
        url = f"{base}/files/{doc_id}?exp={exp}&sig={sig}"
        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat(
            timespec="seconds"
        )

    return {
        "found": True,
        "filename": filename,
        "content_type": mime_for(ctype),
        "bytes": size,
        "url": url,
        "expires_at": expires_at,
        "path": str(blob),
        "url_unavailable_reason": url_unavailable_reason,
    }


@mcp.tool()
def rename_category_tool(old: str, new: str) -> dict[str, Any]:
    """Rename a category globally. If ``new`` already exists, documents merge under it.

    Returns {"affected": N} — the number of documents whose category changed.
    """
    _, conn = _get()
    with _lock:
        n = _rename_category(old, new, conn)
    return {"affected": n}


@mcp.tool()
def delete_category_tool(name: str) -> dict[str, Any]:
    """Unset the category on every document that carries it.

    Returns {"affected": N}.
    """
    _, conn = _get()
    with _lock:
        n = _delete_category(name, conn)
    return {"affected": n}


@mcp.tool()
def rename_tag_tool(old: str, new: str) -> dict[str, Any]:
    """Rename a tag globally. If any document already has ``new``, it's a no-op for that document (merge).

    Returns {"affected": N} — the number of documents where the rename applied.
    """
    _, conn = _get()
    with _lock:
        n = _rename_tag(old, new, conn)
    return {"affected": n}


@mcp.tool()
def delete_tag_tool(name: str) -> dict[str, Any]:
    """Remove a tag from every document that carries it.

    Returns {"affected": N}.
    """
    _, conn = _get()
    with _lock:
        n = _delete_tag(name, conn)
    return {"affected": n}


@mcp.tool()
def suggest_metadata_tool(
    doc_id: str, apply: bool = False
) -> dict[str, Any] | None:
    """Suggest a category + tags for a document via nearest-neighbor voting.

    No LLM — uses the chunk embeddings that already power search. For a
    given document, we average its chunk embeddings, find the K nearest
    documents in the corpus, and vote: categories/tags that appear on a
    large-enough fraction of neighbors become suggestions. Confidence is
    the agreement fraction.

    Inspectable by design: the ``neighbors`` field lists the source
    documents (with similarity) so you can see *why* a suggestion was
    made. Deterministic — same corpus + same doc always yields the same
    suggestion.

    Args:
        doc_id: The document id.
        apply: When True, write the suggested category + tags to the
            document via the same code path as ``update_document_metadata``
            (only fields with non-empty suggestions are written; the
            existing tag list is replaced with the suggestion, so review
            before applying). Default False.

    Returns None if the doc is unknown. Otherwise:
        {
          "doc_id",
          "current":   {"title", "category", "tags"},
          "suggested_category":   str | null,
          "suggested_tags":       [str, ...],
          "category_confidence":  0.0..1.0,
          "tag_confidences":      {tag: 0.0..1.0, ...},
          "neighbors": [
              {"doc_id", "display_title", "similarity", "category", "tags"},
              ...
          ],
          "applied": bool
        }

    Returns an empty suggestion (all fields null/empty, ``neighbors=[]``)
    when the corpus has no labeled neighbors yet — the classifier improves
    as you curate more of the store.
    """
    cfg, conn = _get()
    with _lock:
        s = suggest_metadata(doc_id, conn, cfg)
        if s is None:
            return None
        applied = False
        if apply and (s.suggested_category or s.suggested_tags):
            _update_document_metadata(
                doc_id, conn,
                category=s.suggested_category if s.suggested_category else ...,
                tags=s.suggested_tags if s.suggested_tags else None,
            )
            applied = True

    return _suggestion_payload(s, applied)


def _suggestion_payload(s: Suggestion, applied: bool | None = None) -> dict[str, Any]:
    """Serialize a Suggestion for MCP + API responses."""
    return {
        "doc_id": s.doc_id,
        "current": s.current,
        "suggested_category": s.suggested_category,
        "suggested_tags": list(s.suggested_tags),
        "category_confidence": s.category_confidence,
        "tag_confidences": dict(s.tag_confidences),
        "category_source": s.category_source,
        "tag_source": s.tag_source,
        "anchor_matches": [
            {
                "kind": a.kind,
                "name": a.name,
                "similarity": a.similarity,
                "description": a.description,
            }
            for a in s.anchor_matches
        ],
        "neighbors": [
            {
                "doc_id": n.doc_id,
                "display_title": n.display_title,
                "similarity": n.similarity,
                "category": n.category,
                "tags": list(n.tags),
            }
            for n in s.neighbors
        ],
        "applied": s.applied if applied is None else applied,
    }


def _anchor_payload(a) -> dict[str, Any]:
    return {
        "kind": a.kind,
        "name": a.name,
        "description": a.description,
        "embed_model": a.embed_model,
        "created_at": a.created_at,
        "updated_at": a.updated_at,
    }


@mcp.tool()
def list_anchors_tool() -> list[dict[str, Any]]:
    """List every user-authored label anchor (categories + tags).

    Anchors are natural-language descriptions of what a label means; the
    classifier compares document embeddings against anchor embeddings
    for a zero-shot signal that doesn't depend on existing (possibly
    noisy) document labels.
    """
    _, conn = _get()
    with _lock:
        return [_anchor_payload(a) for a in _anchors.list_anchors(conn)]


@mcp.tool()
def set_anchor_tool(kind: str, name: str, description: str) -> dict[str, Any]:
    """Create or update a label anchor.

    Args:
        kind: "category" or "tag".
        name: Label name (e.g. "bills", "physics").
        description: Short natural-language definition. A vague description
            ("physics stuff") matches everything weakly; a specific one
            ("condensed matter and quantum field theory papers, not
            general science news") matches sharply. Iterate.

    Re-embeds unconditionally on each write. Returns the stored anchor.
    """
    cfg, conn = _get()
    with _lock:
        a = _anchors.set_anchor(conn, cfg, kind, name, description)
    return _anchor_payload(a)


@mcp.tool()
def delete_anchor_tool(kind: str, name: str) -> dict[str, Any]:
    """Delete a label anchor by (kind, name). Idempotent."""
    _, conn = _get()
    with _lock:
        ok = _anchors.delete_anchor(conn, kind, name)
    return {"deleted": ok}


@mcp.tool()
def import_anchors_tool(anchors: list[dict[str, Any]]) -> dict[str, Any]:
    """Bulk-upsert anchors from a list of {kind, name, description} objects.

    Existing anchors with the same (kind, name) are overwritten. Malformed
    items are collected into ``errors`` and don't abort the import.
    """
    cfg, conn = _get()
    with _lock:
        return _anchors.import_anchors(conn, cfg, anchors)


@mcp.tool()
def export_anchors_tool() -> list[dict[str, Any]]:
    """Export every anchor as a portable JSON list — no embeddings, just
    source descriptions. Round-trips through ``import_anchors_tool``."""
    _, conn = _get()
    with _lock:
        return _anchors.export_anchors(conn)


@mcp.tool()
def suggest_metadata_bulk_tool(
    limit: int = 20,
    offset: int = 0,
    missing_only: bool = True,
    apply: bool = False,
) -> dict[str, Any]:
    """Batch-run suggest_metadata across candidate documents.

    Args:
        limit: Max docs to consider (paged with ``offset``).
        offset: Skip the first N candidates.
        missing_only: When True (default), only docs with no category AND
            no tags are considered. Set False to re-verify existing labels
            against embedding neighbors — useful when the corpus is known
            to be inaccurate and you want to spot disagreements.
        apply: When True, write every non-empty suggestion. Off by
            default; review first, apply once you trust the pool.

    Returns:
        {
          "scanned": N,
          "suggestions": [Suggestion, ...],   # docs with actionable output
          "applied":     [doc_id, ...],       # subset written when apply=True
        }

    Suggestion quality is only as good as the labels on nearby documents.
    If the existing taxonomy has errors, batch-applying will amplify them
    — use the doc-detail review flow (or the /suggest-metadata page) to
    curate a clean seed set first.
    """
    cfg, conn = _get()
    with _lock:
        r = suggest_metadata_bulk(
            conn, cfg,
            limit=limit, offset=offset,
            missing_only=missing_only, apply=apply,
        )
    return {
        "scanned": r.scanned,
        "suggestions": [_suggestion_payload(s) for s in r.suggestions],
        "applied": list(r.applied),
    }


def run() -> None:
    """Entry point: run the MCP stdio server."""
    mcp.run()


# ---- Streamable HTTP transport (M7) -----------------------------------------

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate_bind(host: str, auth_token: str | None, no_auth: bool) -> None:
    """Refuse to bind non-loopback interfaces without an auth strategy.

    Loud fail beats silently exposing the corpus. `--no-auth` is an explicit
    opt-out for trusted-network deployments (Tailscale/WireGuard/LAN).
    """
    if host in _LOOPBACK_HOSTS:
        return
    if no_auth:
        return
    if not auth_token:
        raise ValueError(
            f"refusing to bind non-loopback host {host!r} without an auth "
            "token; supply --auth-token-file / ALEXANDRIA_AUTH_TOKEN, or pass "
            "--no-auth to explicitly disable auth"
        )


def _extend_allowed_hosts(hosts: tuple[str, ...]) -> None:
    """Add extra Host: header values to FastMCP's DNS-rebinding allowlist.

    FastMCP defaults to loopback-only (`127.0.0.1:*`, `localhost:*`, `[::1]:*`).
    Any non-loopback deploy (Tailscale, LAN, reverse proxy) must add its own
    hostnames or the middleware returns 421 Misdirected Request before auth
    even runs.
    """
    if not hosts:
        return
    sec = mcp.settings.transport_security
    allowed = list(sec.allowed_hosts)
    origins = list(sec.allowed_origins)
    for h in hosts:
        for pat in (h, f"{h}:*"):
            if pat not in allowed:
                allowed.append(pat)
        for scheme in ("http", "https"):
            origin = f"{scheme}://{h}:*"
            if origin not in origins:
                origins.append(origin)
    sec.allowed_hosts = allowed
    sec.allowed_origins = origins


def build_http_app(
    auth_token: str | None,
    no_auth: bool,
    inner=None,
    allowed_hosts: tuple[str, ...] = (),
    *,
    cfg: Config | None = None,
    conn=None,
    lock: threading.Lock | None = None,
):
    """Return the Streamable HTTP MCP app, wrapped with path-aware auth.

    `inner` lets tests inject a stub Starlette app; production callers omit it
    and get the real FastMCP streamable-http app. The FastMCP singleton's
    session manager can only be started once per process, so tests must not
    reuse the real one.

    When `cfg` is provided and `cfg.web.enabled`, browser-facing `/api/*`
    routes are mounted on the same ASGI app and gated by cookie-or-bearer
    auth. `/mcp` keeps bearer-only for MCP spec compliance.
    """
    from alexandria.web.middleware import SmartAuthMiddleware

    if inner is None:
        _extend_allowed_hosts(allowed_hosts)
        app = mcp.streamable_http_app()
    else:
        app = inner

    session_secret: bytes | None = None
    if cfg is not None and cfg.web.enabled and conn is not None and lock is not None:
        import atexit

        from alexandria.web.api import build_api_routes
        from alexandria.web.auth import get_or_create_session_secret
        from alexandria.web.jobs import JobQueue
        from alexandria.web.pages import build_page_routes

        job_queue = JobQueue(cfg, conn, lock)
        job_queue.start()
        atexit.register(job_queue.stop)

        session_secret = get_or_create_session_secret(cfg)
        for route in build_api_routes(cfg, conn, lock, jobs=job_queue):
            app.router.routes.append(route)
        for route in build_page_routes(
            cfg, session_secret, conn, lock, jobs=job_queue
        ):
            app.router.routes.append(route)

    app.add_middleware(
        SmartAuthMiddleware,
        mcp_token=auth_token,
        session_secret=session_secret,
        no_auth=no_auth,
    )
    return app


def run_http(
    host: str,
    port: int,
    auth_token: str | None,
    no_auth: bool,
    allowed_hosts: tuple[str, ...] = (),
) -> None:
    """Entry point: run the MCP server over Streamable HTTP."""
    validate_bind(host, auth_token, no_auth)
    if not no_auth and not auth_token:
        raise ValueError(
            "no auth token available; set ALEXANDRIA_AUTH_TOKEN, pass "
            "--auth-token-file, configure [network] auth_token_file, or "
            "pass --no-auth"
        )
    import uvicorn

    cfg, conn = _get()
    uvicorn.run(
        build_http_app(
            auth_token, no_auth,
            allowed_hosts=allowed_hosts,
            cfg=cfg, conn=conn, lock=_lock,
        ),
        host=host,
        port=port,
    )


if __name__ == "__main__":
    run()
