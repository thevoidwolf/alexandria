"""Title matching: an opt-in third ranked list, of documents whose title has
every query word. Keyword mode and the fake embedder keep the ranks down to
BM25, so these tests are about the plumbing, not retrieval quality."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from alexandria.common import search_hit_payload
from alexandria.curate import update_document_metadata
from alexandria.ingest import ingest_file
from alexandria.search import search

QUERY = "attention is all you need"


def _ingest(conn, cfg, path: Path, text: str, **kw) -> str:
    path.write_text(text)
    return ingest_file(path, conn, cfg, **kw).doc_id


@pytest.fixture
def library(cfg, conn, tmp_path, fake_embed):
    """A paper titled by its H1, and three short notes citing it by title."""
    paper = _ingest(conn, cfg, tmp_path / "paper.md", "# Attention Is All You Need\n\n"
                    + "We propose a new network architecture based on attention. " * 40)
    citing = [
        _ingest(conn, cfg, tmp_path / f"cite{i}.md", f"# Notes {i}\n\nReferences: Vaswani "
                f"et al. Attention is all you need. NeurIPS 2017. Item {i}.")
        for i in range(3)
    ]
    return cfg, conn, paper, citing


def test_without_titles_the_papers_citing_it_outrank_the_paper(library):
    cfg, conn, paper, citing = library
    hits = search(QUERY, conn, cfg, mode="fts")
    assert hits[0].doc_id in citing
    assert all(h.title_rank is None for h in hits)


def test_a_title_match_lifts_the_paper_to_the_top(library):
    cfg, conn, paper, _ = library
    top = search(QUERY, conn, cfg, mode="fts", titles=True)[0]
    assert top.doc_id == paper
    assert top.title_rank == 1 and "title" in top.matched_in


def test_a_title_needs_every_query_word(library):
    cfg, conn, _, _ = library
    query = QUERY + " tonight"
    with_titles = search(query, conn, cfg, titles=True)
    assert [h.chunk_id for h in with_titles] == [h.chunk_id for h in search(query, conn, cfg)]
    assert all(h.title_rank is None for h in with_titles)


def test_the_credit_goes_to_the_documents_best_candidate_chunk(cfg, conn, tmp_path, fake_embed):
    small = dataclasses.replace(cfg, chunking=dataclasses.replace(cfg.chunking, tokens=30, overlap=0))
    filler = " ".join(["filler"] * 30)
    doc = _ingest(conn, small, tmp_path / "survey.txt",
                  f"{filler} the lunar regolith survey results {filler}")
    update_document_metadata(doc, conn, title="Lunar Regolith Survey")
    credited = [h for h in search("lunar regolith survey", conn, cfg, mode="fts", titles=True)
                if h.title_rank is not None]
    assert len(credited) == 1
    ord_ = conn.execute("SELECT ord FROM chunks WHERE id = ?", (credited[0].chunk_id,)).fetchone()[0]
    assert ord_ == 1        # the chunk the keyword side matched, not the first


def test_a_title_match_with_no_candidate_chunk_brings_its_first_chunk(cfg, conn, tmp_path, fake_embed):
    doc = _ingest(conn, cfg, tmp_path / "x.txt", "nothing about the query in here")
    update_document_metadata(doc, conn, title="Lunar Regolith Survey")
    hits = search("lunar regolith survey", conn, cfg, mode="fts", titles=True)
    assert [h.doc_id for h in hits] == [doc]
    assert hits[0].matched_in == ("title",) and hits[0].title_rank == 1


def test_title_matches_respect_the_category_filter(cfg, conn, tmp_path, fake_embed):
    doc = _ingest(conn, cfg, tmp_path / "x.txt", "nothing relevant", category="elsewhere")
    update_document_metadata(doc, conn, title="Lunar Regolith Survey")
    query = "lunar regolith survey"
    assert search(query, conn, cfg, mode="fts", titles=True, category="here") == []
    assert search(query, conn, cfg, mode="fts", titles=True, category="elsewhere")[0].doc_id == doc


def test_vector_mode_ignores_titles(library):
    cfg, conn, _, _ = library
    assert all(h.title_rank is None for h in search(QUERY, conn, cfg, mode="vec", titles=True))


def test_titles_default_to_the_search_config(library):
    cfg, conn, paper, _ = library
    on = dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, titles=True))
    assert search(QUERY, conn, on, mode="fts")[0].doc_id == paper


def test_the_payload_carries_title_rank(library):
    cfg, conn, _, _ = library
    top = search(QUERY, conn, cfg, mode="fts", titles=True)[0]
    assert search_hit_payload(top)["title_rank"] == 1


def test_title_matches_are_ranked_on_their_titles_alone(cfg, conn, tmp_path, fake_embed):
    wordy = _ingest(conn, cfg, tmp_path / "a.txt", "lunar regolith survey " * 60)
    update_document_metadata(wordy, conn, title="Lunar Regolith Survey of the Northern Craters")
    short = _ingest(conn, cfg, tmp_path / "b.txt", "an unrelated body")
    update_document_metadata(short, conn, title="A Lunar Regolith Survey")
    hits = search("lunar regolith survey", conn, cfg, mode="fts", titles=True)
    by_doc = {h.doc_id: h.title_rank for h in hits if h.title_rank is not None}
    assert by_doc == {short: 1, wordy: 2}    # the shorter title, despite the wordier body
