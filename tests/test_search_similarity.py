"""vec_similarity: the vector side's cosine similarity on each hit. Uses the
fake embedder, whose deterministic unit vectors let the test compute it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import alexandria.search as search_mod
from alexandria.common import search_hit_payload
from alexandria.ingest import ingest_file
from alexandria.search import search


@pytest.fixture
def seeded(cfg, conn, corpus_dir: Path, fake_embed):
    for name in ("plain.txt", "second-bill.txt", "note.md"):
        ingest_file(corpus_dir / name, conn, cfg)
    return cfg, conn


def _vec(cfg, text):
    return search_mod.embed_texts([text], cfg.embeddings)[0]    # the fake embedder


def test_vec_similarity_is_the_cosine_similarity_to_the_query(seeded):
    cfg, conn = seeded
    query = "electricity bill for March"
    hits = search(query, conn, cfg, mode="vec")
    texts = dict(conn.execute("SELECT id, text FROM chunks"))
    assert hits
    for h in hits:
        want = float(np.dot(_vec(cfg, query), _vec(cfg, texts[h.chunk_id])))
        assert h.vec_similarity == pytest.approx(want, abs=1e-5)


def test_a_chunk_searched_by_its_own_text_has_similarity_one(seeded):
    cfg, conn = seeded
    text = conn.execute("SELECT text FROM chunks ORDER BY id LIMIT 1").fetchone()[0]
    assert search(text, conn, cfg, mode="vec")[0].vec_similarity == pytest.approx(1.0, abs=1e-5)


def test_hits_the_vector_side_did_not_return_have_no_similarity(seeded):
    cfg, conn = seeded
    hits = search("bill", conn, cfg, mode="fts")
    assert hits and all(h.vec_similarity is None for h in hits)


def test_search_payload_carries_vec_similarity(seeded):
    cfg, conn = seeded
    h = search("bill", conn, cfg, mode="vec")[0]
    assert search_hit_payload(h)["vec_similarity"] == h.vec_similarity
