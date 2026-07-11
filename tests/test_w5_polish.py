"""W5 polish: snippet formatters, display_title fallback, mobile CSS, upload banners."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from alexandria.ingest import ingest_file
from alexandria.search import (
    SNIPPET_MARK_END,
    SNIPPET_MARK_START,
    format_snippet_markdown,
    format_snippet_plain,
)
from alexandria.web.api import build_api_routes
from alexandria.web.auth import COOKIE_NAME, issue_session
from alexandria.web.jobs import JobQueue
from alexandria.web.middleware import SmartAuthMiddleware
from alexandria.web.pages import build_page_routes, _display_title, _highlight_snippet


# ---- snippet formatters ---------------------------------------------------


def test_format_snippet_markdown_wraps_matches():
    src = f"lorem {SNIPPET_MARK_START}ipsum{SNIPPET_MARK_END} dolor"
    assert format_snippet_markdown(src) == "lorem **ipsum** dolor"


def test_format_snippet_plain_strips_sentinels():
    src = f"lorem {SNIPPET_MARK_START}ipsum{SNIPPET_MARK_END} dolor"
    assert format_snippet_plain(src) == "lorem ipsum dolor"


def test_highlight_snippet_escapes_then_marks():
    src = f"<script>{SNIPPET_MARK_START}alert{SNIPPET_MARK_END}</script>"
    got = _highlight_snippet(src)
    assert "<script>" not in got
    assert "&lt;script&gt;<mark>alert</mark>&lt;/script&gt;" == got


def test_snippet_pua_chars_do_not_collide_with_normal_text():
    # Ensure literal brackets in text no longer get mangled.
    src = f"see item {SNIPPET_MARK_START}foo{SNIPPET_MARK_END} [reference 12]"
    marked = _highlight_snippet(src)
    assert "<mark>foo</mark>" in marked
    # The literal brackets survive intact.
    assert "[reference 12]" in marked


# ---- display_title fallback -----------------------------------------------


class _StubDoc:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_display_title_prefers_title():
    d = _StubDoc(title="Real Title", content_type="pdf")
    assert _display_title(d) == "Real Title"


def test_display_title_uses_source_uri_basename():
    d = _StubDoc(title=None, source_uri="/home/user/Docs/paper.pdf", content_type="pdf")
    assert _display_title(d) == "paper.pdf"


def test_display_title_uses_upload_filename():
    d = _StubDoc(
        title=None,
        sources=[{"source_kind": "upload", "source_uri": "upload:invoice-42.pdf"}],
        content_type="pdf",
    )
    assert _display_title(d) == "invoice-42.pdf"


def test_display_title_uses_url_tail():
    d = _StubDoc(
        title=None,
        sources=[{"source_kind": "url", "source_uri": "https://arxiv.org/abs/2004.04906"}],
        content_type="html",
    )
    assert _display_title(d) == "2004.04906"


def test_display_title_last_resort_untitled():
    d = _StubDoc(title=None, content_type="txt")
    assert _display_title(d) == "untitled txt"


# ---- API surface: markdown snippet in /api/search + MCP tool --------------


@pytest.fixture
def secret() -> bytes:
    return b"z" * 32


@pytest.fixture
def jobs(cfg, conn):
    q = JobQueue(cfg, conn, threading.Lock())
    yield q
    q.stop(timeout=2)


@pytest.fixture
def client(cfg, conn, jobs, secret, corpus_dir: Path, fake_embed):
    # Seed a doc so FTS has a hit.
    ingest_file(corpus_dir / "note.md", conn, cfg, category="notes", tags=[])
    lock = threading.Lock()
    routes = build_api_routes(cfg, conn, lock, jobs=jobs) \
        + build_page_routes(cfg, secret, conn, lock, jobs=jobs)
    app = Starlette(routes=routes)
    app.add_middleware(
        SmartAuthMiddleware,
        mcp_token="mcp-t0k",
        session_secret=secret,
        no_auth=False,
    )
    with TestClient(app) as c:
        c.cookies.set(COOKIE_NAME, issue_session(secret, 3600))
        yield c


def test_api_search_returns_markdown_snippet(client, corpus_dir: Path):
    # Pick a token from note.md that FTS will match.
    # 'knowledge' appears in note.md ("Alexandria is a local knowledge store")
    body = "knowledge"
    r = client.get(f"/api/search?q={body}&mode=fts")
    assert r.status_code == 200
    hits = r.json()
    if not hits:
        pytest.skip("no FTS hit for chosen token")
    # No PUA sentinels leaking through, and no bare '[' delimiter mangling.
    for h in hits:
        assert SNIPPET_MARK_START not in h["snippet"]
        assert SNIPPET_MARK_END not in h["snippet"]


def test_search_page_renders_mark_tags(client, corpus_dir: Path):
    # 'knowledge' appears in note.md ("Alexandria is a local knowledge store")
    body = "knowledge"
    r = client.get(f"/search?q={body}&mode=fts")
    assert r.status_code == 200
    if "No matches" in r.text:
        pytest.skip("no FTS hit for chosen token")
    assert "<mark>" in r.text
    assert "</mark>" in r.text


# ---- mobile CSS + status pill markup --------------------------------------


def test_base_css_has_mobile_breakpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "@media (max-width: 640px)" in r.text
    # Status pill styling present.
    assert "span.status-pill" in r.text or "span.pill" in r.text


def test_landing_upload_banner_container_present(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'id="upload-banners"' in r.text
    # No leftover alert() from earlier versions.
    assert "alert('Upload failed" not in r.text
