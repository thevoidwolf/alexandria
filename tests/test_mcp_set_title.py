"""set_title_tool: correct a document's title over MCP."""
from __future__ import annotations

import pytest

import alexandria.server as srv


@pytest.fixture
def use_cfg(monkeypatch, conn):
    def _use(cfg):
        monkeypatch.setattr(srv, "_get", lambda: (cfg, conn))
    return _use


def test_set_title_tool_sets_a_tidied_title(cfg, use_cfg, fake_embed):
    use_cfg(cfg)
    doc_id = srv.store_document_tool("plain body text", "notes.txt")["doc_id"]
    r = srv.set_title_tool(doc_id, "  Meeting notes,\n March ")
    assert r == {"updated": True, "title": "Meeting notes, March",
                 "display_title": "Meeting notes, March"}
    assert srv.get_document_tool(doc_id)["title"] == "Meeting notes, March"


def test_set_title_tool_blank_falls_back_to_the_filename(cfg, use_cfg, fake_embed):
    use_cfg(cfg)
    doc_id = srv.store_document_tool("# From Its Heading\n\nbody", "note.md")["doc_id"]
    r = srv.set_title_tool(doc_id, "")
    assert r == {"updated": True, "title": None, "display_title": "note.md"}


def test_set_title_tool_unknown_doc(cfg, use_cfg):
    use_cfg(cfg)
    assert srv.set_title_tool("nope", "x") == {"updated": False, "reason": "unknown doc_id"}
