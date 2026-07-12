"""Unit + integration coverage for catalog.display_title.

The helper resolves a human-readable title from three ordered sources:
extractor-recorded title, source-URI basename, then a generic "untitled
<ctype>" fallback. Covers the parsing of all three source-URI flavors
Alexandria records (file paths, http URLs, upload:… synthetic URIs).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from alexandria.catalog import display_title, get_document, list_documents
from alexandria.ingest import ingest_file


# ---- pure helper ----------------------------------------------------------


@pytest.mark.parametrize(
    "title, source_uri, ctype, expected",
    [
        # 1. real title always wins.
        ("Attention Is All You Need", "/x/1706.03762v7.pdf", "pdf",
         "Attention Is All You Need"),
        ("Trim me   ", "/x/foo.pdf", "pdf", "Trim me   "),
        # 2. file-path source → basename.
        (None, "/home/x/docs/The Light of the TARDIS.pdf", "pdf",
         "The Light of the TARDIS.pdf"),
        (None, "C:\\Users\\me\\note.md", "md", "note.md"),
        # 3. http(s) source → last path segment, url-decoded.
        (None, "https://arxiv.org/pdf/1706.03762v7", "pdf", "1706.03762v7"),
        (None, "https://ex.com/a/b/paper%20final.pdf?x=1", "pdf",
         "paper final.pdf"),
        # 4. upload:<name> synthetic URI → strip prefix.
        (None, "upload:invoice-may.pdf", "pdf", "invoice-may.pdf"),
        # 5. empty title, empty basename → untitled fallback.
        (None, None, "pdf", "untitled pdf"),
        ("", "", "html", "untitled html"),
        (None, "/", "txt", "untitled txt"),
    ],
)
def test_display_title_cases(title, source_uri, ctype, expected):
    assert display_title(title, source_uri, ctype) == expected


# ---- integration through the catalog reads --------------------------------


def test_document_row_carries_display_title(cfg, conn, corpus_dir: Path, fake_embed):
    ingest_file(corpus_dir / "plain.txt", conn, cfg, category="notes")
    (doc,) = list_documents(conn)
    # plain.txt has no embedded metadata title → display_title should be the
    # filename, not null or an "untitled" fallback.
    assert doc.display_title == "plain.txt"
    assert doc.title in (None, "")  # nothing extracted

    fat = get_document(doc.id, conn, include_text=True)
    assert fat is not None
    assert fat.display_title == "plain.txt"


def test_search_hit_carries_display_title(cfg, conn, corpus_dir: Path, fake_embed):
    from alexandria.search import search

    ingest_file(corpus_dir / "plain.txt", conn, cfg)
    hits = search("the", conn, cfg, mode="fts", limit=1)
    assert hits
    # No extractor title on a bare .txt → filename-derived display.
    assert hits[0].display_title == "plain.txt"
    assert hits[0].title in (None, "")
