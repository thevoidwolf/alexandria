"""Server-rendered page routes: /, /login, /logout."""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path

from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from alexandria.catalog import get_catalog
from alexandria.config import Config
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

    cookie_max_age = cfg.web.session_max_age_days * 86400

    async def page_index(request: Request) -> Response:
        with lock:
            catalog = get_catalog(conn)
        recent = jobs.list_recent(limit=20) if jobs else []
        return templates.TemplateResponse(request, "index.html", {
            "authenticated": True,
            "catalog": catalog,
            "recent_jobs": recent,
            "max_upload_mb": cfg.web.max_upload_mb,
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

    return [
        Route("/", page_index, methods=["GET"]),
        Route("/login", page_login_get, methods=["GET"]),
        Route("/login", page_login_post, methods=["POST"]),
        Route("/logout", page_logout, methods=["POST"]),
    ]
