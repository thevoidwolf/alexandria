from pathlib import Path

from alexandria.catalog import get_catalog, get_document, list_documents
from alexandria.ingest import ingest_folder


def _seed(cfg, conn, corpus_dir: Path) -> None:
    # Ingest each fixture with distinct category/tags so filters do useful work.
    from alexandria.ingest import ingest_file

    ingest_file(corpus_dir / "plain.txt", conn, cfg,
                category="bills", tags=["utility", "electric"])
    ingest_file(corpus_dir / "second-bill.txt", conn, cfg,
                category="bills", tags=["utility", "water"])
    ingest_file(corpus_dir / "note.md", conn, cfg,
                category="notes", tags=["design"])
    ingest_file(corpus_dir / "paper.md", conn, cfg,
                category="research", tags=["paper"])
    ingest_file(corpus_dir / "page.html", conn, cfg,
                category="web", tags=["reference"])


def test_catalog_reports_totals_and_facets(
    cfg, conn, corpus_dir: Path, fake_embed
):
    _seed(cfg, conn, corpus_dir)
    summary = get_catalog(conn)
    assert summary.total_docs == 5
    assert summary.total_bytes > 0

    cat_counts = {c["name"]: c["count"] for c in summary.categories}
    assert cat_counts == {"bills": 2, "notes": 1, "research": 1, "web": 1}

    tag_counts = {t["name"]: t["count"] for t in summary.tags}
    assert tag_counts["utility"] == 2
    assert tag_counts["water"] == 1


def test_list_documents_filters_by_category(
    cfg, conn, corpus_dir: Path, fake_embed
):
    _seed(cfg, conn, corpus_dir)
    docs = list_documents(conn, category="bills")
    assert len(docs) == 2
    assert all(d.category == "bills" for d in docs)


def test_list_documents_tag_filter_is_AND(
    cfg, conn, corpus_dir: Path, fake_embed
):
    _seed(cfg, conn, corpus_dir)
    # utility+water → only second bill
    docs = list_documents(conn, tags=["utility", "water"])
    assert len(docs) == 1
    assert "water" in docs[0].tags and "utility" in docs[0].tags


def test_list_documents_pagination(cfg, conn, corpus_dir: Path, fake_embed):
    _seed(cfg, conn, corpus_dir)
    first = list_documents(conn, limit=2, offset=0)
    second = list_documents(conn, limit=2, offset=2)
    assert len(first) == 2 and len(second) == 2
    assert {d.id for d in first}.isdisjoint({d.id for d in second})


def test_get_document_with_and_without_text(
    cfg, conn, corpus_dir: Path, fake_embed
):
    _seed(cfg, conn, corpus_dir)
    (any_doc,) = list_documents(conn, category="research")

    slim = get_document(any_doc.id, conn, include_text=False)
    assert slim is not None
    assert slim.extracted_text is None

    fat = get_document(any_doc.id, conn, include_text=True)
    assert fat is not None
    assert fat.extracted_text and len(fat.extracted_text) > 0


def test_get_document_returns_none_for_unknown_id(cfg, conn):
    assert get_document("does-not-exist", conn) is None


def test_get_document_includes_tags_and_sources(
    cfg, conn, corpus_dir: Path, fake_embed
):
    ingest_folder(corpus_dir, conn, cfg, category="mixed", tags=["smoke"])
    docs = list_documents(conn, limit=1)
    doc = get_document(docs[0].id, conn)
    assert doc is not None
    assert "smoke" in doc.tags
    assert len(doc.sources) >= 1
    assert doc.sources[0]["source_kind"] == "file"
