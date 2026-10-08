"""Login throttling, secure cookies, upload/download size limits, folder
roots for the MCP ingest tool, and MCP tool annotations."""
from __future__ import annotations

import dataclasses
import functools
import threading

import anyio
import httpx
import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

import alexandria.server as srv
from alexandria.config import HttpConfig, IngestConfig, NetworkConfig
from alexandria.ingest import fetch as fetch_mod
from alexandria.ingest.fetch import DownloadTooLarge, fetch_url
from alexandria.web.api import build_api_routes
from alexandria.web.auth import (
    COOKIE_NAME,
    LoginThrottle,
    hash_password,
    write_password_hash,
)
from alexandria.web.jobs import JobQueue
from alexandria.web.middleware import SmartAuthMiddleware
from alexandria.web.pages import build_page_routes

SECRET = b"z" * 32


def _app(cfg, conn, jobs=None):
    lock = threading.Lock()
    routes = build_api_routes(cfg, conn, lock, jobs=jobs) \
        + build_page_routes(cfg, SECRET, conn, lock, jobs=jobs)
    return Starlette(routes=routes)


# ---- bearer comparison -----------------------------------------------------


def test_bearer_rejected_when_no_token_configured():
    app = Starlette()
    app.add_middleware(
        SmartAuthMiddleware, mcp_token=None, session_secret=None, no_auth=False
    )
    with TestClient(app) as c:
        r = c.post("/mcp", headers={"Authorization": "Bearer anything"})
    assert r.status_code == 401


# ---- login throttle --------------------------------------------------------


def test_throttle_blocks_after_max_failures_and_ages_out():
    t = LoginThrottle(max_failures=3, window_seconds=60)
    for i in range(3):
        assert t.retry_after("ip", now=1000 + i) == 0
        t.record_failure("ip", now=1000 + i)
    assert t.retry_after("ip", now=1003) > 0
    assert t.retry_after("other", now=1003) == 0
    # Oldest failure (t=1000) ages out at t=1060.
    assert t.retry_after("ip", now=1061) == 0


def test_throttle_reset_clears_failures():
    t = LoginThrottle(max_failures=1, window_seconds=60)
    t.record_failure("ip", now=0)
    assert t.retry_after("ip", now=1) > 0
    t.reset("ip")
    assert t.retry_after("ip", now=1) == 0


def test_login_returns_429_after_repeated_failures(cfg, conn, monkeypatch):
    write_password_hash(cfg.web_password_hash_path, hash_password("pw", rounds=4))
    with TestClient(_app(cfg, conn)) as c:
        for _ in range(10):
            assert c.post("/login", data={"password": "nope"}).status_code == 401

        # Once throttled, bcrypt isn't even consulted — and the right
        # password is refused too.
        def boom(*a, **kw):
            raise AssertionError("verify_password called while throttled")

        monkeypatch.setattr("alexandria.web.pages.verify_password", boom)
        r = c.post("/login", data={"password": "pw"}, follow_redirects=False)
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) > 0


def test_successful_login_resets_throttle(cfg, conn):
    write_password_hash(cfg.web_password_hash_path, hash_password("pw", rounds=4))
    with TestClient(_app(cfg, conn)) as c:
        for _ in range(9):
            c.post("/login", data={"password": "nope"})
        r = c.post("/login", data={"password": "pw"}, follow_redirects=False)
        assert r.status_code == 303
        for _ in range(9):
            assert c.post("/login", data={"password": "nope"}).status_code == 401


# ---- secure cookie ---------------------------------------------------------


def _login_cookie_header(cfg, conn) -> str:
    write_password_hash(cfg.web_password_hash_path, hash_password("pw", rounds=4))
    with TestClient(_app(cfg, conn)) as c:
        r = c.post("/login", data={"password": "pw"}, follow_redirects=False)
    return r.headers["set-cookie"]


def test_cookie_not_secure_on_plain_http(cfg, conn):
    header = _login_cookie_header(cfg, conn)
    assert header.startswith(COOKIE_NAME + "=")
    assert "secure" not in header.lower()


def test_cookie_secure_when_public_url_is_https(cfg, conn):
    cfg = dataclasses.replace(
        cfg, network=NetworkConfig(public_base_url="https://alx.example.com")
    )
    assert "secure" in _login_cookie_header(cfg, conn).lower()


# ---- upload Content-Length -------------------------------------------------


@pytest.fixture
def jobs(cfg, conn):
    q = JobQueue(cfg, conn, threading.Lock())
    yield q
    q.stop(timeout=2)


def test_upload_requires_content_length(cfg, conn, jobs):
    with TestClient(_app(cfg, conn, jobs)) as c:
        # A generator body goes out chunked, with no Content-Length.
        r = c.post(
            "/api/upload",
            content=iter([b"--x\r\n"]),
            headers={"content-type": "multipart/form-data; boundary=x"},
        )
    assert r.status_code == 411


def test_upload_rejects_malformed_content_length(cfg, conn, jobs):
    status: list[int] = []

    async def send(msg):
        if msg["type"] == "http.response.start":
            status.append(msg["status"])

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http", "method": "POST", "path": "/api/upload",
        "headers": [(b"content-length", b"abc"),
                    (b"content-type", b"multipart/form-data; boundary=x")],
        "query_string": b"", "server": ("t", 80), "client": ("c", 1),
        "scheme": "http", "root_path": "", "http_version": "1.1",
    }
    anyio.run(_app(cfg, conn, jobs), scope, receive, send)
    assert status == [400]


# ---- download cap ----------------------------------------------------------


def _patch_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        fetch_mod.httpx, "Client", functools.partial(httpx.Client, transport=transport)
    )


HTTP = HttpConfig(respect_robots_txt=False, max_download_mb=1)


def test_fetch_rejects_declared_oversize(monkeypatch):
    _patch_transport(monkeypatch, lambda req: httpx.Response(
        200, headers={"content-length": str(5 * 1024 * 1024)}, content=b"x"
    ))
    with pytest.raises(DownloadTooLarge):
        fetch_url("https://example.com/big", HTTP)


def test_fetch_rejects_undeclared_oversize_stream(monkeypatch):
    def handler(req):
        chunks = (b"x" * 65536 for _ in range(32))  # 2 MiB, no length header
        return httpx.Response(200, content=chunks)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(DownloadTooLarge):
        fetch_url("https://example.com/stream", HTTP)


def test_fetch_under_cap_succeeds(monkeypatch):
    _patch_transport(monkeypatch, lambda req: httpx.Response(
        200, headers={"content-type": "text/plain"}, content=b"hello"
    ))
    f = fetch_url("https://example.com/ok", HTTP)
    assert f.data == b"hello"
    assert f.content_type_hint == "text/plain"


def test_fetch_error_status_wins_over_size(monkeypatch):
    _patch_transport(monkeypatch, lambda req: httpx.Response(
        404, headers={"content-length": str(5 * 1024 * 1024)}, content=b"x"
    ))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_url("https://example.com/missing", HTTP)


def test_oversized_robots_txt_is_ignored(monkeypatch):
    fetch_mod._ROBOTS_CACHE.clear()

    def handler(req):
        if req.url.path == "/robots.txt":
            return httpx.Response(200, content=b"Disallow: /\n" * 100_000)
        return httpx.Response(200, content=b"page")

    _patch_transport(monkeypatch, handler)
    f = fetch_url("https://robots.example/page", dataclasses.replace(
        HTTP, respect_robots_txt=True
    ))
    assert f.data == b"page"
    fetch_mod._ROBOTS_CACHE.clear()


# ---- ingest_folder_tool roots ----------------------------------------------


@pytest.fixture
def use_cfg(monkeypatch, conn):
    def _use(cfg):
        monkeypatch.setattr(srv, "_get", lambda: (cfg, conn))
    return _use


def test_folder_tool_refuses_without_roots(cfg, use_cfg, corpus_dir):
    use_cfg(cfg)
    with pytest.raises(PermissionError, match="disabled: no .ingest. allowed_roots"):
        srv.ingest_folder_tool(str(corpus_dir))


def test_folder_tool_refuses_path_outside_roots(cfg, use_cfg, corpus_dir, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    use_cfg(dataclasses.replace(cfg, ingest=IngestConfig(allowed_roots=(str(allowed),))))
    with pytest.raises(PermissionError, match="outside"):
        srv.ingest_folder_tool(str(corpus_dir))
    # ".." can't climb out either.
    with pytest.raises(PermissionError):
        srv.ingest_folder_tool(str(allowed / ".." / corpus_dir.name))


def test_folder_tool_ingests_inside_roots_but_not_symlink_escapes(
    cfg, use_cfg, corpus_dir, tmp_path, fake_embed
):
    secret = tmp_path / "outside" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("private key material")
    (corpus_dir / "link.txt").symlink_to(secret)

    use_cfg(dataclasses.replace(
        cfg, ingest=IngestConfig(allowed_roots=(str(corpus_dir),))
    ))
    r = srv.ingest_folder_tool(str(corpus_dir), glob="*.txt")
    assert r["ingested"]
    escaped = [e for e in r["errors"] if e["path"].endswith("link.txt")]
    assert escaped and "outside" in escaped[0]["reason"]


# ---- tool annotations ------------------------------------------------------


def test_every_tool_is_annotated():
    for tool in srv.mcp._tool_manager.list_tools():
        assert tool.annotations is not None, tool.name


@pytest.mark.parametrize("name", [
    "delete_document_tool", "delete_category_tool", "delete_tag_tool",
    "rename_category_tool", "rename_tag_tool", "suggest_metadata_bulk_tool",
    "import_anchors_tool", "delete_anchor_tool",
])
def test_destructive_tools_marked(name):
    a = srv.mcp._tool_manager.get_tool(name).annotations
    assert a.destructiveHint is True and a.readOnlyHint is False


@pytest.mark.parametrize("name", [
    "search_tool", "get_document_tool", "list_documents_tool",
    "get_catalog_tool", "get_original_tool",
])
def test_read_tools_marked_read_only(name):
    assert srv.mcp._tool_manager.get_tool(name).annotations.readOnlyHint is True
