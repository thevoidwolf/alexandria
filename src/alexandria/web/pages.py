"""Server-rendered page routes: /, /login, /logout, /search, /documents{,/id}."""
from __future__ import annotations

import html
import sqlite3
import threading
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from urllib.parse import urlencode

from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from alexandria.catalog import display_title, get_catalog, get_document, list_documents
from alexandria.config import Config
from alexandria.search import SNIPPET_MARK_END, SNIPPET_MARK_START, search
from alexandria.web.auth import (
    COOKIE_NAME,
    issue_session,
    read_password_hash,
    verify_password,
)
from alexandria.web.jobs import JobQueue

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _version() -> str:
    try:
        return importlib_metadata.version("alexandria")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0+dev"


def _human_bytes(n: int) -> str:
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < step:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} PB"


def _describe_job(j) -> str:
    if j.kind == "upload":
        return j.input.get("filename") or "upload"
    if j.kind == "url":
        return j.input.get("url") or "url"
    return j.kind


def _display_title(d) -> str:
    """Jinja shim over catalog.display_title.

    Prefer the pre-computed ``display_title`` field (SearchHit + DocumentRow
    both carry it). Falls back to computing on the fly for any bare object
    the templates might still hand us.
    """
    pre = getattr(d, "display_title", None)
    if pre:
        return pre
    title = getattr(d, "title", None)
    uri = getattr(d, "source_uri", None)
    if not uri:
        sources = getattr(d, "sources", None)
        if sources:
            first = sources[0]
            uri = first.get("source_uri") if isinstance(first, dict) else getattr(first, "source_uri", None)
    return display_title(title, uri, getattr(d, "content_type", "doc"))


def _highlight_snippet(s: str) -> str:
    """Turn PUA snippet sentinels into <mark> tags.

    Escape first, then swap the sentinels. Since the sentinels are PUA chars
    they can't appear in the source text, so this is lossless.
    """
    escaped = html.escape(s)
    return (
        escaped
        .replace(SNIPPET_MARK_START, "<mark>")
        .replace(SNIPPET_MARK_END, "</mark>")
    )


def _split_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _job_detail(j) -> str:
    if j.status == "error":
        return j.error or ""
    if j.status == "done" and j.result:
        if j.result.get("was_duplicate"):
            return f"duplicate ({j.result.get('dedup_gate') or '?'})"
        bits: list[str] = []
        if j.result.get("title"):
            bits.append(f'"{j.result["title"]}"')
        if j.result.get("n_chunks"):
            bits.append(f"{j.result['n_chunks']} chunks")
        if j.result.get("suggested_metadata"):
            sm = j.result["suggested_metadata"]
            hint_bits: list[str] = []
            if sm.get("suggested_category"):
                hint_bits.append(sm["suggested_category"])
            n_tags = len(sm.get("suggested_tags") or [])
            if n_tags:
                hint_bits.append(f"{n_tags} tag{'s' if n_tags != 1 else ''}")
            if hint_bits:
                bits.append("→ suggest " + " · ".join(hint_bits))
        return " · ".join(bits)
    return ""


def build_page_routes(
    cfg: Config,
    session_secret: bytes,
    conn: sqlite3.Connection,
    lock: threading.Lock,
    jobs: JobQueue | None,
) -> list[Route]:
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
    templates.env.globals["title"] = cfg.web.title
    templates.env.globals["version"] = _version()
    templates.env.globals["home"] = str(cfg.home)
    templates.env.globals["human_bytes"] = _human_bytes
    templates.env.globals["describe_job"] = _describe_job
    templates.env.globals["job_detail"] = _job_detail
    templates.env.globals["highlight_snippet"] = _highlight_snippet
    templates.env.globals["display_title"] = _display_title

    cookie_max_age = cfg.web.session_max_age_days * 86400

    async def page_index(request: Request) -> Response:
        with lock:
            catalog = get_catalog(conn)
        recent = jobs.list_recent(limit=20) if jobs else []
        return templates.TemplateResponse(request, "index.html", {
            "authenticated": True,
            "nav": "home",
            "catalog": catalog,
            "recent_jobs": recent,
            "max_upload_mb": cfg.web.max_upload_mb,
        })

    async def page_search(request: Request) -> Response:
        params = request.query_params
        q = (params.get("q") or "").strip()
        category = params.get("category") or None
        tags_raw = params.get("tags") or ""
        tags = _split_tags(tags_raw)
        mode = params.get("mode", "hybrid")
        if mode not in ("hybrid", "fts", "vec"):
            mode = "hybrid"

        with lock:
            catalog = get_catalog(conn)
            hits = search(
                q, conn, cfg,
                category=category, tags=tags, limit=25, mode=mode,  # type: ignore[arg-type]
            ) if q else []

        return templates.TemplateResponse(request, "search.html", {
            "authenticated": True,
            "nav": "search",
            "catalog": catalog,
            "q": q,
            "category": category,
            "tags": tags_raw,
            "mode": mode,
            "hits": hits,
        })

    async def page_list_documents(request: Request) -> Response:
        params = request.query_params
        category = params.get("category") or None
        tags_raw = params.get("tags") or ""
        tags = _split_tags(tags_raw)
        try:
            offset = max(0, int(params.get("offset", "0")))
        except ValueError:
            offset = 0
        limit = 20

        with lock:
            catalog = get_catalog(conn)
            # Over-fetch one row to detect whether a next page exists.
            docs = list_documents(
                conn, category=category, tags=tags,
                limit=limit + 1, offset=offset,
            )
        has_next = len(docs) > limit
        docs = docs[:limit]

        def page_url(new_offset: int) -> str:
            qs = {"offset": new_offset}
            if category:
                qs["category"] = category
            if tags_raw:
                qs["tags"] = tags_raw
            return "/documents?" + urlencode(qs)

        return templates.TemplateResponse(request, "documents.html", {
            "authenticated": True,
            "nav": "documents",
            "catalog": catalog,
            "docs": docs,
            "category": category,
            "tags": tags_raw,
            "offset": offset,
            "prev_offset": offset - limit if offset > 0 else None,
            "next_offset": offset + limit if has_next else None,
            "page_url": page_url,
        })

    async def page_document_detail(request: Request) -> Response:
        doc_id = request.path_params["doc_id"]
        with lock:
            doc = get_document(doc_id, conn, include_text=False)
            catalog = get_catalog(conn) if doc else None
        if doc is None:
            return templates.TemplateResponse(request, "document.html", {
                "authenticated": True,
                "nav": "documents",
                "doc": None,
            }, status_code=404)
        return templates.TemplateResponse(request, "document.html", {
            "authenticated": True,
            "nav": "documents",
            "doc": doc,
            "catalog": catalog,
        })

    async def page_login_get(request: Request) -> Response:
        return templates.TemplateResponse(request, "login.html", {
            "authenticated": False,
            "error": None,
        })

    async def page_login_post(request: Request) -> Response:
        form = await request.form()
        password = str(form.get("password") or "")

        stored = read_password_hash(cfg.web_password_hash_path)
        if stored is None:
            return templates.TemplateResponse(request, "login.html", {
                "authenticated": False,
                "error": "No web password set. Run `alexandria set-web-password` on the server.",
            }, status_code=503)

        if not password or not verify_password(password, stored):
            return templates.TemplateResponse(request, "login.html", {
                "authenticated": False,
                "error": "Invalid password.",
            }, status_code=401)

        cookie = issue_session(session_secret, cookie_max_age)
        resp = RedirectResponse(url="/", status_code=303)
        resp.set_cookie(
            COOKIE_NAME, cookie,
            max_age=cookie_max_age,
            httponly=True,
            samesite="strict",
            path="/",
        )
        return resp

    async def page_logout(_request: Request) -> Response:
        resp = RedirectResponse(url="/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME, path="/")
        return resp

    async def page_taxonomy(request: Request) -> Response:
        with lock:
            catalog = get_catalog(conn)
        return templates.TemplateResponse(request, "taxonomy.html", {
            "authenticated": True,
            "nav": "taxonomy",
            "catalog": catalog,
        })

    async def page_suggest_metadata(request: Request) -> Response:
        return templates.TemplateResponse(request, "suggest_metadata.html", {
            "authenticated": True,
            "nav": "suggest",
        })

    async def page_anchors(request: Request) -> Response:
        return templates.TemplateResponse(request, "anchors.html", {
            "authenticated": True,
            "nav": "anchors",
        })

    return [
        Route("/", page_index, methods=["GET"]),
        Route("/login", page_login_get, methods=["GET"]),
        Route("/login", page_login_post, methods=["POST"]),
        Route("/logout", page_logout, methods=["POST"]),
        Route("/search", page_search, methods=["GET"]),
        Route("/documents", page_list_documents, methods=["GET"]),
        Route("/documents/{doc_id}", page_document_detail, methods=["GET"]),
        Route("/taxonomy", page_taxonomy, methods=["GET"]),
        Route("/suggest-metadata", page_suggest_metadata, methods=["GET"]),
        Route("/anchors", page_anchors, methods=["GET"]),
    ]
