"""Backend curate operations: delete_document, rename/delete categories & tags."""
from __future__ import annotations

from pathlib import Path

import pytest

from alexandria.curate import (
    delete_category,
    delete_document,
    delete_tag,
    rename_category,
    rename_tag,
    update_document_metadata,
)
from alexandria.ingest import ingest_file


@pytest.fixture
def seeded(cfg, conn, corpus_dir: Path, fake_embed):
    """Five docs across mixed categories/tags."""
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
    return cfg, conn


# ---- delete_document ------------------------------------------------------


def test_delete_document_cascades(seeded):
    cfg, conn = seeded
    doc_id, sha = conn.execute(
        "SELECT id, sha256_raw FROM documents WHERE category='notes'"
    ).fetchone()

    blob = cfg.blobs_dir / sha[:2] / sha
    assert blob.exists()

    assert delete_document(doc_id, conn, cfg) is True

    assert conn.execute("SELECT 1 FROM documents WHERE id=?", (doc_id,)).fetchone() is None
    assert conn.execute("SELECT 1 FROM chunks WHERE doc_id=?", (doc_id,)).fetchone() is None
    assert conn.execute("SELECT 1 FROM tags WHERE doc_id=?", (doc_id,)).fetchone() is None
    assert conn.execute("SELECT 1 FROM document_sources WHERE doc_id=?", (doc_id,)).fetchone() is None
    assert conn.execute("SELECT 1 FROM documents_fts WHERE doc_id=?", (doc_id,)).fetchone() is None
    assert conn.execute("SELECT 1 FROM chunks_fts WHERE doc_id=?", (doc_id,)).fetchone() is None
    assert not blob.exists()


def test_delete_document_unknown_returns_false(cfg, conn):
    assert delete_document("does-not-exist", conn, cfg) is False


def test_delete_document_search_no_longer_returns_it(seeded):
    from alexandria.search import search
    cfg, conn = seeded
    doc_id = conn.execute(
        "SELECT id FROM documents WHERE category='research'"
    ).fetchone()[0]

    hits_before = [h.doc_id for h in search("Alexandria", conn, cfg, mode="fts")]
    delete_document(doc_id, conn, cfg)
    hits_after = [h.doc_id for h in search("Alexandria", conn, cfg, mode="fts")]

    assert doc_id not in hits_after
    assert set(hits_after).issubset(set(hits_before))


# ---- update_document_metadata --------------------------------------------


def test_update_metadata_replaces_tags(seeded):
    _, conn = seeded
    doc_id = conn.execute(
        "SELECT id FROM documents WHERE category='notes'"
    ).fetchone()[0]
    assert update_document_metadata(
        doc_id, conn, category="idea", tags=["fresh", "draft"]
    ) is True

    row = conn.execute("SELECT category FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert row[0] == "idea"
    tags = sorted(r[0] for r in conn.execute("SELECT tag FROM tags WHERE doc_id=?", (doc_id,)))
    assert tags == ["draft", "fresh"]


def test_update_metadata_partial_leaves_absent_fields(seeded):
    _, conn = seeded
    doc_id = conn.execute(
        "SELECT id FROM documents WHERE category='notes'"
    ).fetchone()[0]
    original_tags = sorted(r[0] for r in conn.execute(
        "SELECT tag FROM tags WHERE doc_id=?", (doc_id,)))

    # Only category, no tags key.
    update_document_metadata(doc_id, conn, category="misc")
    new_tags = sorted(r[0] for r in conn.execute(
        "SELECT tag FROM tags WHERE doc_id=?", (doc_id,)))
    assert new_tags == original_tags
    cat = conn.execute("SELECT category FROM documents WHERE id=?", (doc_id,)).fetchone()[0]
    assert cat == "misc"


def test_update_metadata_null_category_unsets(seeded):
    _, conn = seeded
    doc_id = conn.execute(
        "SELECT id FROM documents WHERE category='notes'"
    ).fetchone()[0]
    update_document_metadata(doc_id, conn, category=None)
    row = conn.execute("SELECT category FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert row[0] is None


def test_update_metadata_unknown_doc(cfg, conn):
    assert update_document_metadata("nope", conn, category="x") is False


# ---- category rename / delete --------------------------------------------


def test_rename_category(seeded):
    _, conn = seeded
    assert rename_category("bills", "invoices", conn) == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM documents WHERE category='invoices'"
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM documents WHERE category='bills'"
    ).fetchone()[0] == 0


def test_rename_category_merges_into_existing(seeded):
    _, conn = seeded
    # 2 bills + 1 research → all end up as 'archive'
    rename_category("bills", "archive", conn)
    rename_category("research", "archive", conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM documents WHERE category='archive'"
    ).fetchone()[0] == 3


def test_rename_category_rejects_blank(seeded):
    _, conn = seeded
    with pytest.raises(ValueError):
        rename_category("bills", "   ", conn)


def test_delete_category_unsets(seeded):
    _, conn = seeded
    assert delete_category("bills", conn) == 2
    remaining = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE category IS NULL"
    ).fetchone()[0]
    assert remaining == 2


# ---- tag rename / delete -------------------------------------------------


def test_rename_tag(seeded):
    _, conn = seeded
    assert rename_tag("utility", "recurring", conn) == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM tags WHERE tag='recurring'"
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM tags WHERE tag='utility'"
    ).fetchone()[0] == 0


def test_rename_tag_merges_where_target_exists(seeded):
    _, conn = seeded
    # 'water' is on second-bill; rename 'utility' → 'water' should keep
    # second-bill with 'water' once (dedup) and add 'water' to plain.txt.
    n = rename_tag("utility", "water", conn)
    assert n == 2
    rows = conn.execute(
        "SELECT doc_id FROM tags WHERE tag='water'"
    ).fetchall()
    # both bills should now have 'water'
    assert len(rows) == 2


def test_rename_tag_no_op_when_names_equal(seeded):
    _, conn = seeded
    assert rename_tag("utility", "utility", conn) == 0


def test_rename_tag_rejects_blank(seeded):
    _, conn = seeded
    with pytest.raises(ValueError):
        rename_tag("utility", "", conn)


def test_delete_tag_removes_from_all(seeded):
    _, conn = seeded
    assert delete_tag("utility", conn) == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM tags WHERE tag='utility'"
    ).fetchone()[0] == 0
