"""Nearest-neighbor tag/category suggestion.

Real embeddings (via ``fake_embed`` is *not* semantic — but here we use
real ones so vector similarity actually reflects meaning). Corpus is a
handful of fixture docs with distinct topic families so the k-NN can
group them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from alexandria.classify import Suggestion, suggest_metadata
from alexandria.ingest import ingest_file


@pytest.fixture
def bills_corpus(cfg, conn, corpus_dir: Path):
    """Two labeled bills + a third unlabeled bill-like doc.

    The unlabeled doc is the target for suggestion; the two labeled ones
    are the neighbors whose (category, tags) should propagate.
    """
    ingest_file(corpus_dir / "plain.txt", conn, cfg,
                category="bills", tags=["utility", "electric"])
    ingest_file(corpus_dir / "second-bill.txt", conn, cfg,
                category="bills", tags=["utility", "water"])
    # Add a third bill-shaped doc without any labels.
    unlabeled_path = corpus_dir / "third-bill.txt"
    unlabeled_path.write_text(
        "Account statement. Amount due for services this billing period. "
        "Please remit payment by the due date shown. Utility charges apply.\n"
    )
    r = ingest_file(unlabeled_path, conn, cfg)
    return cfg, conn, r.doc_id


def test_suggests_majority_category(bills_corpus):
    cfg, conn, target = bills_corpus
    s = suggest_metadata(target, conn, cfg)
    assert s is not None
    assert s.suggested_category == "bills"
    assert s.category_confidence >= 0.5  # both neighbors are bills


def test_suggests_shared_tag_only(bills_corpus):
    cfg, conn, target = bills_corpus
    s = suggest_metadata(target, conn, cfg)
    assert s is not None
    # 'utility' is on both neighbors → 100% → suggested.
    assert "utility" in s.suggested_tags
    # 'electric' is only on one → 50% → suggested at default threshold 0.35.
    # 'water' is only on one → same.
    for tag in s.suggested_tags:
        assert tag in {"utility", "electric", "water"}


def test_neighbors_are_inspectable(bills_corpus):
    cfg, conn, target = bills_corpus
    s = suggest_metadata(target, conn, cfg)
    assert s is not None
    assert 1 <= len(s.neighbors) <= 15
    for n in s.neighbors:
        assert 0.0 < n.similarity <= 1.0
        assert n.doc_id != target                # self-exclusion
        assert n.display_title                   # never null
    # Similarities are sorted descending.
    sims = [n.similarity for n in s.neighbors]
    assert sims == sorted(sims, reverse=True)


def test_returns_current_metadata_for_diffing(bills_corpus):
    cfg, conn, target = bills_corpus
    s = suggest_metadata(target, conn, cfg)
    assert s is not None
    # Target was ingested without a category or tags.
    assert s.current["category"] is None
    assert s.current["tags"] == []


def test_unknown_doc_returns_none(cfg, conn):
    assert suggest_metadata("no-such-id", conn, cfg) is None


def test_empty_neighbor_pool_returns_empty_suggestion(cfg, conn, corpus_dir: Path):
    """One doc in the corpus → no neighbors → empty suggestion, not a crash."""
    r = ingest_file(corpus_dir / "plain.txt", conn, cfg,
                    category="notes", tags=["misc"])
    s = suggest_metadata(r.doc_id, conn, cfg)
    assert s is not None
    assert s.suggested_category is None
    assert s.suggested_tags == []
    assert s.neighbors == []


def test_threshold_filters_low_agreement_tags(bills_corpus, cfg):
    """Raising tag_min_fraction to 0.9 keeps only tags on ~all neighbors."""
    from dataclasses import replace
    _, conn, target = bills_corpus
    strict = replace(cfg, classify=replace(cfg.classify, tag_min_fraction=0.9))
    s = suggest_metadata(target, conn, strict)
    assert s is not None
    # 'utility' is on both neighbors (100%); 'electric'/'water' on one (50%).
    assert s.suggested_tags == ["utility"]


def test_apply_via_mcp_writes_metadata(bills_corpus):
    """The MCP tool applies suggestions via update_document_metadata."""
    from alexandria import server as srv

    cfg, conn, target = bills_corpus
    # Pin module state the tool reads.
    srv._cfg = cfg
    srv._conn = conn

    r = srv.suggest_metadata_tool(target, apply=True)
    assert r is not None
    assert r["applied"] is True

    # Persisted?
    row = conn.execute(
        "SELECT category FROM documents WHERE id = ?", (target,)
    ).fetchone()
    assert row[0] == "bills"
    tags = [t[0] for t in conn.execute(
        "SELECT tag FROM tags WHERE doc_id = ?", (target,)
    )]
    assert "utility" in tags
