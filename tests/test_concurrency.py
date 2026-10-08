"""The DB lock must not be held across slow work, and nothing slow may run on
the event loop: either would stall every other request on the HTTP server."""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from starlette.applications import Starlette
from starlette.testclient import TestClient

import alexandria.ingest.orchestrator as orch
from alexandria.ingest.fetch import fetch_file
from alexandria.ingest.orchestrator import ingest_fetched
from alexandria.web.api import build_api_routes


def test_ingest_releases_lock_during_extract_and_embed(
    cfg, conn, corpus_dir: Path, fake_embed, monkeypatch
):
    lock = threading.Lock()
    seen: dict[str, bool] = {}

    real_extract = orch.extract
    real_embed = orch.embed_texts

    def spy_extract(*a, **kw):
        seen["extract"] = lock.locked()
        return real_extract(*a, **kw)

    def spy_embed(*a, **kw):
        seen["embed"] = lock.locked()
        return real_embed(*a, **kw)

    monkeypatch.setattr(orch, "extract", spy_extract)
    monkeypatch.setattr(orch, "embed_texts", spy_embed)

    r = ingest_fetched(fetch_file(corpus_dir / "plain.txt"), conn, cfg, lock=lock)
    assert not r.was_duplicate
    assert seen == {"extract": False, "embed": False}
    assert not lock.locked()


def test_ingest_rechecks_dedup_after_unlocked_phase(
    cfg, conn, corpus_dir: Path, fake_embed, monkeypatch
):
    """Same bytes stored by someone else while we embedded → duplicate, not
    a UNIQUE violation."""
    lock = threading.Lock()
    copy = corpus_dir / "plain-copy.txt"
    copy.write_bytes((corpus_dir / "plain.txt").read_bytes())

    real_embed = orch.embed_texts
    raced = False

    def racing_embed(texts, emb_cfg):
        nonlocal raced
        if not raced:
            raced = True
            ingest_fetched(fetch_file(copy), conn, cfg, lock=lock)
        return real_embed(texts, emb_cfg)

    monkeypatch.setattr(orch, "embed_texts", racing_embed)

    r = ingest_fetched(fetch_file(corpus_dir / "plain.txt"), conn, cfg, lock=lock)
    assert r.was_duplicate
    assert r.dedup_gate == "raw"
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM document_sources").fetchone()[0] == 2


def test_api_does_not_block_event_loop_on_db_lock(cfg, conn):
    lock = threading.Lock()
    app = Starlette(routes=build_api_routes(cfg, conn, lock))
    with TestClient(app) as client:
        lock.acquire()  # simulate a long ingest holding the DB
        try:
            waiting = threading.Thread(
                target=client.get, args=("/api/catalog",), daemon=True
            )
            waiting.start()

            # /api/info takes no lock; it must still be served while the
            # catalog request waits. A blocked event loop would hang here.
            result: dict = {}
            other = threading.Thread(
                target=lambda: result.update(r=client.get("/api/info")),
                daemon=True,
            )
            other.start()
            other.join(timeout=10)
            assert not other.is_alive(), "event loop blocked behind DB lock"
            assert result["r"].status_code == 200
            assert waiting.is_alive()
        finally:
            lock.release()
        waiting.join(timeout=10)
        assert not waiting.is_alive()


def test_mcp_tools_run_off_the_event_loop(cfg, conn, monkeypatch):
    import alexandria.server as srv

    monkeypatch.setattr(srv, "_get", lambda: (cfg, conn))

    tool = srv.mcp._tool_manager.get_tool("get_catalog_tool")
    assert tool is not None and tool.is_async

    # Registered schema still reflects the wrapped function's signature.
    search = srv.mcp._tool_manager.get_tool("search_tool")
    assert {"query", "category", "tags", "limit", "mode"} <= set(
        search.parameters["properties"]
    )

    loop_thread: dict = {}
    real = srv.get_catalog

    def spy(c):
        loop_thread["tool"] = threading.get_ident()
        return real(c)

    monkeypatch.setattr(srv, "get_catalog", spy)

    async def go():
        loop_thread["loop"] = threading.get_ident()
        return await srv.mcp.call_tool("get_catalog_tool", {})

    asyncio.run(go())
    assert loop_thread["tool"] != loop_thread["loop"]
