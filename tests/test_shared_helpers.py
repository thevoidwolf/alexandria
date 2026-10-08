"""Shared serializers, the robots.txt cache, and server lazy init."""
from __future__ import annotations

import functools
import threading
import time
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

import alexandria.server as srv
from alexandria.common import split_tags
from alexandria.config import HttpConfig
from alexandria.ingest import fetch as fetch_mod
from alexandria.ingest import ingest_file
from alexandria.web.api import build_api_routes


def test_split_tags():
    assert split_tags(None) == []
    assert split_tags("") == []
    assert split_tags(" a, ,b ,c") == ["a", "b", "c"]


def test_mcp_and_http_search_payloads_match(cfg, conn, corpus_dir: Path, fake_embed, monkeypatch):
    """One serializer now backs both; guard against them drifting apart."""
    ingest_file(corpus_dir / "plain.txt", conn, cfg, category="bills", tags=["utility"])
    ingest_file(corpus_dir / "note.md", conn, cfg, category="notes")
    monkeypatch.setattr(srv, "_get", lambda: (cfg, conn))

    mcp_hits = srv.search_tool("electric bill", limit=5)
    app = Starlette(routes=build_api_routes(cfg, conn, threading.Lock()))
    with TestClient(app) as c:
        http_hits = c.get("/api/search", params={"q": "electric bill", "limit": 5}).json()
    assert mcp_hits and mcp_hits == http_hits


# ---- robots.txt cache --------------------------------------------------------


@pytest.fixture
def robots_server(monkeypatch):
    fetch_mod._ROBOTS_CACHE.clear()
    calls: list[str] = []

    def handler(req):
        calls.append(str(req.url))
        if req.url.path == "/robots.txt":
            return httpx.Response(200, content=b"User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b"page")

    monkeypatch.setattr(
        fetch_mod.httpx, "Client",
        functools.partial(httpx.Client, transport=httpx.MockTransport(handler)),
    )
    yield calls
    fetch_mod._ROBOTS_CACHE.clear()


HTTP = HttpConfig(respect_robots_txt=True)


def _robots_fetches(calls, host="example.com"):
    return sum(1 for u in calls if u == f"https://{host}/robots.txt")


def test_robots_cached_within_ttl(robots_server):
    fetch_mod.fetch_url("https://example.com/a", HTTP)
    fetch_mod.fetch_url("https://example.com/b", HTTP)
    assert _robots_fetches(robots_server) == 1


def test_robots_refetched_after_ttl(robots_server, monkeypatch):
    fetch_mod.fetch_url("https://example.com/a", HTTP)
    later = time.monotonic() + fetch_mod._ROBOTS_TTL_SECONDS + 1
    monkeypatch.setattr(fetch_mod.time, "monotonic", lambda: later)
    fetch_mod.fetch_url("https://example.com/b", HTTP)
    assert _robots_fetches(robots_server) == 2


def test_robots_cache_is_bounded(robots_server, monkeypatch):
    monkeypatch.setattr(fetch_mod, "_ROBOTS_CACHE_MAX", 3)
    for i in range(5):
        fetch_mod.fetch_url(f"https://h{i}.example/x", HTTP)
    assert list(fetch_mod._ROBOTS_CACHE) == [
        "https://h2.example", "https://h3.example", "https://h4.example"
    ]


# ---- server lazy init --------------------------------------------------------


def test_get_opens_one_connection_under_concurrency(cfg, monkeypatch):
    monkeypatch.setattr(srv, "_cfg", None)
    monkeypatch.setattr(srv, "_conn", None)
    monkeypatch.setattr(srv, "load_config", lambda: cfg)
    opened: list[object] = []

    def slow_connect(path, dim):
        time.sleep(0.05)  # widen the race window
        obj = object()
        opened.append(obj)
        return obj

    monkeypatch.setattr(srv, "connect", slow_connect)
    barrier = threading.Barrier(8)
    results: list[object] = []

    def worker():
        barrier.wait()
        results.append(srv._get()[1])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(opened) == 1
    assert all(r is opened[0] for r in results)
