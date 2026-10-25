"""Opt-in search parameters (evals variants). Each defaults to today's behaviour,
which the other search tests pin; these pin what the non-default values do.
Uses the fake embedder: this is about query plumbing, not ranking."""
from __future__ import annotations

from pathlib import Path

import pytest

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


def test_quote_terms_can_join_with_or():
    assert _quote_terms('half-life of "C++"', " OR ") == '"half-life" OR "of" OR """C++"""'
    assert _quote_terms("  ... !! ", " OR ") is None


def test_or_matches_a_chunk_with_any_of_the_words(seeded):
    cfg, conn = seeded
    hits = search("templates xyzzynothing", conn, cfg, mode="fts", fts_operator="or")
    assert _titles(hits) == {"physics.txt"}


def test_or_does_not_parse_fts5_syntax(seeded):
    # By default this is valid syntax, a phrase, and the two words aren't
    # adjacent. Joined with OR, every whitespace-separated word is a literal
    # term, quotes and all, so either word is enough.
    cfg, conn = seeded
    assert search('"half carbon"', conn, cfg, mode="fts") == []
    hits = search('"half carbon"', conn, cfg, mode="fts", fts_operator="or")
    assert _titles(hits) == {"physics.txt"}


def test_or_with_no_words_returns_nothing(seeded):
    cfg, conn = seeded
    assert search("--- ???", conn, cfg, mode="fts", fts_operator="or") == []


def test_unknown_fts_operator_is_refused(seeded):
    cfg, conn = seeded
    with pytest.raises(ValueError, match="fts_operator"):
        search("templates", conn, cfg, fts_operator="near")  # type: ignore[arg-type]


@pytest.mark.parametrize("rrf_k", [None, 20])
def test_hybrid_scores_use_the_given_rrf_k(seeded, rrf_k):
    cfg, conn = seeded
    kwargs = {} if rrf_k is None else {"rrf_k": rrf_k}
    k = 60 if rrf_k is None else rrf_k
    hits = search("half-life templates", conn, cfg, fts_operator="or", **kwargs)
    assert hits and any(h.fts_rank and h.vec_rank for h in hits)
    for h in hits:
        expected = sum(1.0 / (k + r) for r in (h.fts_rank, h.vec_rank) if r)
        assert h.score == pytest.approx(expected)
