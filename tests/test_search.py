"""Search tests. These use the *real* embedding model so retrieval quality assertions
are meaningful. The first test in a session pays the model load cost (~5s);
subsequent tests reuse the cached model via embed.py's lru_cache.
"""
from pathlib import Path

import pytest

from alexandria.ingest import ingest_file
from alexandria.search import search


@pytest.fixture
def seeded(cfg, conn, corpus_dir: Path):
    ingest_file(corpus_dir / "plain.txt", conn, cfg,
                category="bills", tags=["utility", "electric"])
    ingest_file(corpus_dir / "second-bill.txt", conn, cfg,
                category="bills", tags=["utility", "water"])
    ingest_file(corpus_dir / "note.md", conn, cfg,
                category="notes", tags=["design"])
    ingest_file(corpus_dir / "paper.md", conn, cfg,
                category="research", tags=["paper", "retrieval"])
    ingest_file(corpus_dir / "page.html", conn, cfg,
                category="web", tags=["reference"])
    return cfg, conn


def test_empty_query_returns_empty(seeded):
    cfg, conn = seeded
    assert search("", conn, cfg) == []
    assert search("   ", conn, cfg) == []


def test_fts_mode_finds_exact_token(seeded):
    from alexandria.search import SNIPPET_MARK_END, SNIPPET_MARK_START

    cfg, conn = seeded
    hits = search("BM25", conn, cfg, mode="fts")
    assert hits, "FTS should match BM25"
    # Snippet from FTS wraps the matched term in the PUA sentinels.
    marked = f"{SNIPPET_MARK_START}BM25{SNIPPET_MARK_END}"
    assert any(marked in h.snippet for h in hits)


def test_vec_mode_handles_paraphrase(seeded):
    cfg, conn = seeded
    hits = search("combining lexical and dense retrieval", conn, cfg, mode="vec", limit=3)
    top_ids = [h.doc_id for h in hits[:2]]
    # paper.md is the ground-truth doc for this concept
    paper_ids = {r[0] for r in conn.execute(
        "SELECT id FROM documents WHERE category = 'research'"
    )}
    assert paper_ids & set(top_ids), "paper should surface in top 2 for paraphrase"


def test_hybrid_boosts_docs_both_sides_agree_on(seeded):
    cfg, conn = seeded
    hits = search("account amount due", conn, cfg, limit=5)
    # The water-bill doc contains "account", "amount", and "due" (FTS-friendly)
    # AND is semantically about a bill (vec-friendly). It should rank near the top.
    top_categories = [h.category for h in hits[:2]]
    assert "bills" in top_categories


def test_category_filter_restricts_results(seeded):
    cfg, conn = seeded
    hits = search("account", conn, cfg, category="bills", limit=10)
    assert hits
    assert all(h.category == "bills" for h in hits)


def test_tag_filter_is_AND(seeded):
    cfg, conn = seeded
    hits = search("account", conn, cfg, tags=["utility", "water"], limit=10)
    assert len(hits) == 1
    assert "water" in hits[0].tags and "utility" in hits[0].tags


def test_impossible_filter_returns_no_results(seeded):
    cfg, conn = seeded
    hits = search("account", conn, cfg, tags=["utility", "does-not-exist"], limit=10)
    assert hits == []


def test_hits_carry_source_uri(seeded):
    cfg, conn = seeded
    hits = search("BM25", conn, cfg, mode="fts", limit=1)
    assert hits
    assert hits[0].source_uri and hits[0].source_uri.endswith((".md", ".html"))


def test_limit_is_respected(seeded):
    cfg, conn = seeded
    hits = search("the", conn, cfg, limit=2)
    assert len(hits) <= 2


# ---- rank signals (fts_rank / vec_rank / matched_in) -----------------------


def test_fts_mode_populates_only_fts_rank(seeded):
    cfg, conn = seeded
    hits = search("BM25", conn, cfg, mode="fts", limit=3)
    assert hits
    for h in hits:
        assert h.fts_rank is not None and h.fts_rank >= 1
        assert h.vec_rank is None
        assert h.matched_in == ("fts",)
    # Ranks are strictly increasing along the returned order.
    ranks = [h.fts_rank for h in hits]
    assert ranks == sorted(ranks)


def test_vec_mode_populates_only_vec_rank(seeded):
    cfg, conn = seeded
    hits = search("payment for services", conn, cfg, mode="vec", limit=3)
    assert hits
    for h in hits:
        assert h.vec_rank is not None and h.vec_rank >= 1
        assert h.fts_rank is None
        assert h.matched_in == ("vec",)


def test_hybrid_marks_dual_matched_hits(seeded):
    cfg, conn = seeded
    # A query with both distinctive tokens and clear semantics — a bill-domain
    # doc should surface in both retrievers' windows.
    hits = search("account amount due", conn, cfg, mode="hybrid", limit=10)
    assert hits
    both = [h for h in hits if h.matched_in == ("fts", "vec")]
    assert both, "at least one hit should appear in both retriever lists"
    # For dual-matched hits, both ranks are populated with sensible integers.
    for h in both:
        assert h.fts_rank is not None and h.fts_rank >= 1
        assert h.vec_rank is not None and h.vec_rank >= 1
