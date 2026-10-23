"""Eval indexes: one throwaway Alexandria home per (corpus, settings), cached."""
from dataclasses import replace
from pathlib import Path

import pytest

from evals import index as ev_index
from evals.datasets import Dataset
from evals.index import (
    IndexSettings,
    build_index,
    canonical_ids,
    index_key,
    open_index,
    translate_qrels,
)

SMALL = IndexSettings(chunk_tokens=50, chunk_overlap=10)
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def model_rev(monkeypatch):
    """Stand in for the Hub lookup; tests that care set ``rev["value"]``."""
    rev = {"value": "rev-1"}
    monkeypatch.setattr(ev_index, "model_revision", lambda name: rev["value"])
    return rev


def _dataset(**docs: str) -> Dataset:
    return Dataset(name="tiny", docs=docs, queries={}, qrels={})


@pytest.fixture
def tiny() -> Dataset:
    return _dataset(
        a="Reciprocal rank fusion merges keyword and vector rankings.",
        b="Reciprocal rank fusion merges keyword and vector rankings.",
        c="The water bill for March was unusually high.",
    )


def test_build_maps_every_doc_and_collapses_identical_ones(tiny, tmp_path, fake_embed):
    built = build_index(tiny, SMALL, tmp_path / "cache")
    assert set(built.doc_map) == {"a", "b", "c"}
    assert built.doc_map["a"] == built.doc_map["b"] != built.doc_map["c"]
    assert built.stats["n_docs"] == 2
    assert built.stats["n_collapsed"] == 1


def test_built_index_is_searchable_through_open_index(tiny, tmp_path, fake_embed):
    from alexandria.search import search

    built = build_index(tiny, SMALL, tmp_path / "cache")
    cfg, conn = open_index(built)
    try:
        hits = search("water bill", conn, cfg, mode="fts")
    finally:
        conn.close()
    assert [h.doc_id for h in hits] == [built.doc_map["c"]]
    assert cfg.chunking.tokens == 50


def test_open_index_applies_query_side_overrides(tiny, tmp_path, fake_embed):
    built = build_index(tiny, SMALL, tmp_path / "cache")
    cfg, conn = open_index(built, query_instruction="Represent: ")
    conn.close()
    assert cfg.embeddings.query_instruction == "Represent: "


def test_second_build_reuses_the_cache(tiny, tmp_path, fake_embed, monkeypatch):
    first = build_index(tiny, SMALL, tmp_path / "cache")

    def boom(*a, **kw):
        raise AssertionError("cached index was rebuilt")

    monkeypatch.setattr(ev_index, "ingest_fetched", boom)
    second = build_index(tiny, SMALL, tmp_path / "cache")
    assert second.doc_map == first.doc_map
    assert second.home == first.home


def test_unfinished_build_is_discarded(tiny, tmp_path, fake_embed):
    cache = tmp_path / "cache"
    partial = cache / index_key(tiny, SMALL)
    (partial / "home").mkdir(parents=True)
    (partial / "home" / "leftover").write_text("from a crashed run")

    built = build_index(tiny, SMALL, cache)
    assert not (built.home / "leftover").exists()
    assert set(built.doc_map) == {"a", "b", "c"}


def test_key_changes_with_settings_and_corpus(tiny):
    assert index_key(tiny, SMALL) == index_key(tiny, IndexSettings(chunk_tokens=50, chunk_overlap=10))
    assert index_key(tiny, SMALL) != index_key(tiny, IndexSettings(chunk_tokens=40, chunk_overlap=10))
    assert index_key(tiny, SMALL) != index_key(
        tiny, IndexSettings(embed_model="BAAI/bge-base-en-v1.5", embed_dim=768,
                            chunk_tokens=50, chunk_overlap=10)
    )
    edited = _dataset(**{**tiny.docs, "c": "The water bill for April was normal."})
    assert index_key(tiny, SMALL) != index_key(edited, SMALL)


def test_index_never_lands_in_the_users_home(tiny, tmp_path, fake_embed, monkeypatch):
    real_home = tmp_path / "real-alexandria"
    real_home.mkdir()
    monkeypatch.setenv("ALEXANDRIA_HOME", str(real_home))

    built = build_index(tiny, SMALL, tmp_path / "cache")
    assert built.home.is_relative_to(tmp_path / "cache")
    assert list(real_home.iterdir()) == []


def test_canonical_id_is_the_first_dataset_id_per_document():
    assert canonical_ids({"b": "X", "a": "X", "c": "Y"}) == {"X": "a", "Y": "c"}


def test_translate_qrels_merges_collapsed_docs_with_the_max_grade():
    doc_map = {"a": "X", "b": "X", "c": "Y"}
    qrels = {"q1": {"b": 1, "c": 0}, "q2": {"a": 1, "b": 2}}
    assert translate_qrels(qrels, doc_map) == {"q1": {"a": 1, "c": 0}, "q2": {"a": 2}}


def test_translate_qrels_keeps_docs_missing_from_the_index():
    assert translate_qrels({"q": {"gone": 1}}, {"a": "X"}) == {"q": {"gone": 1}}


def test_home_holds_no_blobs(tiny, tmp_path, fake_embed):
    built = build_index(tiny, SMALL, tmp_path / "cache")
    assert [p for p in (built.home / "blobs").rglob("*") if p.is_file()] == []


def test_key_changes_when_the_model_revision_moves(tiny, model_rev):
    before = index_key(tiny, SMALL)
    model_rev["value"] = "rev-2"
    assert index_key(tiny, SMALL) != before


def test_build_records_the_model_revision(tiny, tmp_path, fake_embed, model_rev):
    model_rev["value"] = "5c38ec7"
    assert build_index(tiny, SMALL, tmp_path / "cache").stats["embed_revision"] == "5c38ec7"


def test_byte_documents_go_in_through_the_upload_path_as_pdfs(tmp_path, fake_embed):
    ds = Dataset(
        name="pdfs", docs={"p1": (FIXTURES / "paper.pdf").read_bytes()},
        queries={}, qrels={}, doc_suffix=".pdf",
    )
    built = build_index(ds, SMALL, tmp_path / "cache")
    cfg, conn = open_index(built)
    try:
        content_type, extractor = conn.execute(
            "SELECT content_type, extractor FROM documents").fetchone()
        source = conn.execute("SELECT source_kind, source_uri FROM document_sources").fetchone()
    finally:
        conn.close()
    assert content_type == "pdf"
    assert extractor.startswith("pypdf")
    assert source == ("upload", "upload:p1.pdf")


def test_pdf_backend_is_pinned_in_the_index_config(tiny, tmp_path, fake_embed):
    built = build_index(tiny, replace(SMALL, pdf_backend="auto"), tmp_path / "cache")
    cfg, conn = open_index(built)
    conn.close()
    assert cfg.extractors.pdf.backend == "auto"
    assert index_key(tiny, SMALL) != index_key(tiny, replace(SMALL, pdf_backend="auto"))
