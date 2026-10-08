"""Search must not silently drop results for ordinary-looking queries or big
filters. Uses the fake embedder: these are about query plumbing, not ranking."""
from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path

import pytest

import alexandria.search as search_mod
from alexandria.config import ChunkingConfig, EmbeddingsConfig
from alexandria.ingest import ingest_file
from alexandria.search import _quote_terms, search


@pytest.fixture
def seeded(cfg, conn, corpus_dir: Path, fake_embed):
    (corpus_dir / "physics.txt").write_text(
        "The half-life of carbon-14 is about 5,730 years. "
        "Unrelated aside: C++ templates and what's-new notes."
    )
    ingest_file(corpus_dir / "physics.txt", conn, cfg, category="science")
    ingest_file(corpus_dir / "second-bill.txt", conn, cfg, category="bills")
    return cfg, conn


def _titles(hits) -> set[str]:
    return {Path(h.source_uri).name for h in hits}


@pytest.mark.parametrize("query, expected", [
    ("8891-3320-2", "second-bill.txt"),   # the README's own fts example
    ("half-life", "physics.txt"),
    ("carbon-14", "physics.txt"),
    ("what's", "physics.txt"),
    ("C++ templates", "physics.txt"),
    ("5,730 years", "physics.txt"),
])
def test_punctuated_queries_still_match(seeded, query, expected):
    cfg, conn = seeded
    hits = search(query, conn, cfg, mode="fts")
    assert expected in _titles(hits), query


@pytest.mark.parametrize("query", [
    '"half life"',               # phrase
    "carb*",                     # prefix
    "xyzzynothing OR templates",  # OR
    "NEAR(half carbon, 5)",
])
def test_fts5_syntax_still_honoured(seeded, query):
    cfg, conn = seeded
    assert "physics.txt" in _titles(search(query, conn, cfg, mode="fts"))


def test_or_is_not_broadened_by_fallback(seeded):
    # Valid syntax runs as written: an AND of two terms where only one is
    # present must not match.
    cfg, conn = seeded
    assert search("templates xyzzynothing", conn, cfg, mode="fts") == []


def test_punctuation_only_query_returns_empty(seeded):
    cfg, conn = seeded
    assert search("--- ??? ()", conn, cfg, mode="fts") == []


def test_quote_terms():
    assert _quote_terms('half-life of "C++"') == '"half-life" "of" """C++"""'
    assert _quote_terms("  ... !! ") is None


def test_large_category_filter_beyond_parameter_limit(cfg, conn, corpus_dir, fake_embed):
    """Filters used to inline every eligible chunk rowid as a bound
    parameter, which fails past SQLITE_LIMIT_VARIABLE_NUMBER (32766 in
    default builds). Simulate a small limit."""
    cfg = dataclasses.replace(cfg, chunking=ChunkingConfig(tokens=10, overlap=0))
    big = corpus_dir / "big.txt"
    big.write_text(" ".join(f"word{i} zebra" for i in range(1000)))
    ingest_file(big, conn, cfg, category="big")
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert n_chunks > 100

    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 50)
    try:
        for mode in ("fts", "vec", "hybrid"):
            hits = search("zebra", conn, cfg, category="big", mode=mode, limit=5)
            assert len(hits) == 5, mode
            assert all(h.category == "big" for h in hits)
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 32766)


def test_filter_matching_nothing_returns_empty(seeded):
    cfg, conn = seeded
    for mode in ("fts", "vec", "hybrid"):
        assert search("half-life", conn, cfg, category="nope", mode=mode) == []
        assert search("half-life", conn, cfg, tags=["nope"], mode=mode) == []


def test_query_instruction_prefixes_query_only(seeded, monkeypatch):
    cfg, conn = seeded
    cfg = dataclasses.replace(
        cfg, embeddings=EmbeddingsConfig(query_instruction="Represent: ")
    )
    seen: list[list[str]] = []
    real = search_mod.embed_texts

    def spy(texts, emb_cfg):
        seen.append(list(texts))
        return real(texts, emb_cfg)

    monkeypatch.setattr(search_mod, "embed_texts", spy)
    search("carbon", conn, cfg, mode="vec")
    assert seen == [["Represent: carbon"]]
