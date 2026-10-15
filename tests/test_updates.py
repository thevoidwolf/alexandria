"""Changed content at a known file/URL source updates that document in place,
and content from any source replaces a document named by ``target_doc_id``."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from alexandria.ingest import (
    ContentConflict,
    DocumentNotFound,
    SourceConflict,
    ingest_file,
)
from alexandria.ingest.fetch import Fetched
from alexandria.ingest.hash import sha256_bytes
from alexandria.ingest.orchestrator import ingest_fetched
from alexandria.originals import blob_path, lookup_blob_for
from alexandria.search import search

ORIGINAL = "# A\n\nalpha wolves\n"


def _write(p: Path, text: str) -> Path:
    p.write_bytes(text.encode())
    return p


def _append(p: Path, text: str) -> None:
    p.write_bytes(p.read_bytes() + text.encode())


def _doc(conn, doc_id: str):
    return conn.execute(
        "SELECT sha256_raw, extracted_text, category FROM documents WHERE id = ?",
        (doc_id,),
    ).fetchone()


def _sources(conn, doc_id: str) -> list[str]:
    return sorted(r[0] for r in conn.execute(
        "SELECT source_uri FROM document_sources WHERE doc_id = ?", (doc_id,)
    ))


def _tags(conn, doc_id: str) -> list[str]:
    return sorted(r[0] for r in conn.execute(
        "SELECT tag FROM tags WHERE doc_id = ?", (doc_id,)
    ))


def _count(conn, sql: str, *args) -> int:
    return conn.execute(sql, args).fetchone()[0]


def test_same_path_edit_updates_in_place(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    r1 = ingest_file(a, conn, cfg, category="notes", tags=["x"])
    old_sha = sha256_bytes(a.read_bytes())

    _append(a, "\nbeta ravens\n")
    r2 = ingest_file(a, conn, cfg, tags=["y"])

    assert r2.was_updated is True
    assert r2.was_duplicate is False
    assert r2.dedup_gate is None
    assert r2.doc_id == r1.doc_id
    assert _count(conn, "SELECT COUNT(*) FROM documents") == 1
    assert _doc(conn, r1.doc_id)[2] == "notes"
    assert _tags(conn, r1.doc_id) == ["x", "y"]
    assert r1.doc_id in {h.doc_id for h in search("ravens", conn, cfg, mode="fts")}
    fts_rows = conn.execute(
        "SELECT extracted_text FROM documents_fts WHERE doc_id = ?", (r1.doc_id,)
    ).fetchall()
    assert len(fts_rows) == 1
    assert "ravens" in fts_rows[0][0]
    assert _count(conn, "SELECT COUNT(*) FROM chunks_vec") == _count(
        conn, "SELECT COUNT(*) FROM chunks"
    )
    assert not blob_path(cfg, old_sha).exists()
    assert blob_path(cfg, sha256_bytes(a.read_bytes())).exists()


def test_same_path_edit_keeps_category_even_if_another_is_passed(
    cfg, conn, tmp_path, fake_embed
):
    a = _write(tmp_path / "a.md", ORIGINAL)
    r1 = ingest_file(a, conn, cfg, category="notes", tags=["x"])

    _append(a, "\nbeta ravens\n")
    r2 = ingest_file(a, conn, cfg, category="vaults", tags=["y"])

    assert r2.was_updated is True
    assert _doc(conn, r1.doc_id)[2] == "notes"


def test_rerun_after_update_is_source_duplicate(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    ingest_file(a, conn, cfg)
    _append(a, "\nbeta ravens\n")
    assert ingest_file(a, conn, cfg).was_updated is True

    r3 = ingest_file(a, conn, cfg)

    assert r3.was_duplicate is True
    assert r3.dedup_gate == "source"
    assert r3.was_updated is False


def test_whitespace_only_edit_is_text_duplicate(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    r1 = ingest_file(a, conn, cfg)
    before = _doc(conn, r1.doc_id)[1]

    a.write_bytes(b"# A  \r\n\r\nalpha wolves   \r\n")
    r2 = ingest_file(a, conn, cfg)

    assert r2.was_duplicate is True
    assert r2.dedup_gate == "text"
    assert r2.was_updated is False
    assert _doc(conn, r1.doc_id)[1] == before


def test_whitespace_only_edit_refreshes_raw_so_next_scan_skips_extract(
    cfg, conn, tmp_path, fake_embed, monkeypatch
):
    a = _write(tmp_path / "a.md", ORIGINAL)
    r1 = ingest_file(a, conn, cfg)
    old_sha = sha256_bytes(a.read_bytes())

    a.write_bytes(b"# A  \r\n\r\nalpha wolves   \r\n")
    new_sha = sha256_bytes(a.read_bytes())
    r2 = ingest_file(a, conn, cfg)

    assert r2.dedup_gate == "text"
    assert _doc(conn, r1.doc_id)[0] == new_sha
    assert blob_path(cfg, new_sha).exists()
    assert not blob_path(cfg, old_sha).exists()

    def no_extract(*args, **kwargs):
        raise AssertionError("extract should not run for an unchanged file")

    monkeypatch.setattr("alexandria.ingest.orchestrator.extract", no_extract)
    assert ingest_file(a, conn, cfg).dedup_gate == "source"


def test_text_identical_alias_does_not_refresh_raw(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    b = tmp_path / "b.md"
    b.write_bytes(ORIGINAL.replace("\n", "\r\n").encode())
    r1 = ingest_file(a, conn, cfg)
    rb = ingest_file(b, conn, cfg)
    assert (rb.dedup_gate, rb.doc_id) == ("text", r1.doc_id)
    assert _sources(conn, r1.doc_id) == sorted([str(a), str(b)])
    raw_before = _doc(conn, r1.doc_id)[0]

    a.write_bytes(b"# A  \n\nalpha wolves   \n")
    r = ingest_file(a, conn, cfg)

    assert r.dedup_gate == "text"
    assert _doc(conn, r1.doc_id)[0] == raw_before


def test_url_source_change_updates(cfg, conn, fake_embed):
    r1 = ingest_fetched(
        Fetched(b"v1 text", "url", "https://ex.com/a", "text/plain"), conn, cfg
    )
    r2 = ingest_fetched(
        Fetched(b"v2 other text", "url", "https://ex.com/a", "text/plain"), conn, cfg
    )

    assert r2.was_updated is True
    assert r2.doc_id == r1.doc_id


def test_update_detaches_alias_sources(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    b = _write(tmp_path / "b.md", ORIGINAL)
    r1 = ingest_file(a, conn, cfg)
    assert ingest_file(b, conn, cfg).doc_id == r1.doc_id
    original_text = _doc(conn, r1.doc_id)[1]

    _append(a, "\nbeta ravens\n")
    assert ingest_file(a, conn, cfg).was_updated is True
    assert _sources(conn, r1.doc_id) == [str(a)]

    rb = ingest_file(b, conn, cfg)
    assert rb.doc_id != r1.doc_id
    assert rb.was_duplicate is False
    assert _doc(conn, rb.doc_id)[1] == original_text


def test_edit_into_copy_of_other_doc_moves_source(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    b = _write(tmp_path / "b.md", "# B\n\ngamma owls\n")
    ra = ingest_file(a, conn, cfg)
    rb = ingest_file(b, conn, cfg)
    a_text = _doc(conn, ra.doc_id)[1]

    a.write_bytes(b.read_bytes())
    r = ingest_file(a, conn, cfg)

    assert r.was_duplicate is True
    assert r.dedup_gate == "raw"
    assert r.doc_id == rb.doc_id
    assert str(a) in _sources(conn, rb.doc_id)
    assert _doc(conn, ra.doc_id)[1] == a_text
    assert _sources(conn, ra.doc_id) == []


def test_failed_extract_leaves_doc_untouched(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    r1 = ingest_file(a, conn, cfg)
    before = (_doc(conn, r1.doc_id), _sources(conn, r1.doc_id))

    a.write_bytes(b"")
    with pytest.raises(ValueError, match="No extractable text"):
        ingest_file(a, conn, cfg)

    assert (_doc(conn, r1.doc_id), _sources(conn, r1.doc_id)) == before


def test_failed_write_rolls_back_update(
    cfg, conn, tmp_path, fake_embed, monkeypatch
):
    a = _write(tmp_path / "a.md", ORIGINAL)
    r1 = ingest_file(a, conn, cfg)
    old_sha = sha256_bytes(a.read_bytes())

    def snapshot():
        return (
            _doc(conn, r1.doc_id)[:2],
            _count(conn, "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", r1.doc_id),
            _count(conn, "SELECT COUNT(*) FROM chunks_vec"),
            _sources(conn, r1.doc_id),
        )

    before = snapshot()

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("alexandria.ingest.orchestrator.insert_chunks", boom)
    _append(a, "\nbeta ravens\n")
    with pytest.raises(RuntimeError, match="disk full"):
        ingest_file(a, conn, cfg)

    assert snapshot() == before
    assert blob_path(cfg, old_sha).exists()
    assert not blob_path(cfg, sha256_bytes(a.read_bytes())).exists()


def test_update_without_kept_blobs(cfg, conn, tmp_path, fake_embed):
    cfg = dataclasses.replace(
        cfg, storage=dataclasses.replace(cfg.storage, keep_blobs=False)
    )
    a = _write(tmp_path / "a.md", ORIGINAL)
    ingest_file(a, conn, cfg)

    _append(a, "\nbeta ravens\n")
    assert ingest_file(a, conn, cfg).was_updated is True


def test_update_when_old_blob_missing(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    ingest_file(a, conn, cfg)
    blob_path(cfg, sha256_bytes(a.read_bytes())).unlink()

    _append(a, "\nbeta ravens\n")
    r2 = ingest_file(a, conn, cfg)

    assert r2.was_updated is True
    assert blob_path(cfg, sha256_bytes(a.read_bytes())).exists()


def test_uploads_with_same_filename_are_separate_docs(cfg, conn, fake_embed):
    r1 = ingest_fetched(Fetched(b"first notes", "upload", "upload:notes.md"), conn, cfg)
    r2 = ingest_fetched(Fetched(b"second notes", "upload", "upload:notes.md"), conn, cfg)

    assert r1.doc_id != r2.doc_id
    assert r1.was_duplicate is False
    assert r2.was_duplicate is False


def test_upload_same_bytes_is_raw_duplicate(cfg, conn, fake_embed):
    ingest_fetched(Fetched(b"same notes", "upload", "upload:notes.md"), conn, cfg)
    r2 = ingest_fetched(Fetched(b"same notes", "upload", "upload:notes.md"), conn, cfg)

    assert r2.was_duplicate is True
    assert r2.dedup_gate == "raw"


def test_explicit_update_from_new_path(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    b = _write(tmp_path / "b.md", "# B\n\ngamma owls\n")
    ra = ingest_file(a, conn, cfg)

    r = ingest_file(b, conn, cfg, target_doc_id=ra.doc_id)

    assert r.was_updated is True
    assert r.was_duplicate is False
    assert r.doc_id == ra.doc_id
    assert _count(conn, "SELECT COUNT(*) FROM documents") == 1
    assert _sources(conn, ra.doc_id) == [str(b)]
    assert "gamma owls" in _doc(conn, ra.doc_id)[1]
    assert "alpha wolves" not in _doc(conn, ra.doc_id)[1]


def test_explicit_update_with_category_replaces_it(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    b = _write(tmp_path / "b.md", "# B\n\ngamma owls\n")
    ra = ingest_file(a, conn, cfg, category="notes")

    r = ingest_file(b, conn, cfg, category="archive", target_doc_id=ra.doc_id)

    assert r.was_updated is True
    assert _doc(conn, ra.doc_id)[2] == "archive"


def test_explicit_update_unchanged_content(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    c = _write(tmp_path / "c.md", ORIGINAL)
    ra = ingest_file(a, conn, cfg, category="notes")

    r = ingest_file(c, conn, cfg, category="archive", target_doc_id=ra.doc_id)

    assert r.was_duplicate is True
    assert r.dedup_gate == "raw"
    assert r.was_updated is False
    assert r.doc_id == ra.doc_id
    assert _sources(conn, ra.doc_id) == sorted([str(a), str(c)])
    assert _doc(conn, ra.doc_id)[2] == "notes"


def test_explicit_update_text_identical_keeps_target_bytes(
    cfg, conn, tmp_path, fake_embed
):
    a = _write(tmp_path / "a.md", ORIGINAL)
    c = tmp_path / "c.md"
    c.write_bytes(ORIGINAL.replace("\n", "\r\n").encode())
    ra = ingest_file(a, conn, cfg)
    raw_before = _doc(conn, ra.doc_id)[0]

    r = ingest_file(c, conn, cfg, target_doc_id=ra.doc_id)

    assert (r.was_duplicate, r.dedup_gate, r.doc_id) == (True, "text", ra.doc_id)
    assert _doc(conn, ra.doc_id)[0] == raw_before
    assert _sources(conn, ra.doc_id) == sorted([str(a), str(c)])


def test_explicit_update_conflicting_content_refused(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    b = _write(tmp_path / "b.md", "# B\n\ngamma owls\n")
    ra = ingest_file(a, conn, cfg)
    rb = ingest_file(b, conn, cfg)
    c = tmp_path / "c.md"
    c.write_bytes(b.read_bytes())

    def snapshot():
        return (
            _doc(conn, ra.doc_id), _doc(conn, rb.doc_id),
            _sources(conn, ra.doc_id), _sources(conn, rb.doc_id),
            _count(conn, "SELECT COUNT(*) FROM documents"),
            _count(conn, "SELECT COUNT(*) FROM chunks"),
            _count(conn, "SELECT COUNT(*) FROM document_sources"),
        )

    before = snapshot()
    with pytest.raises(ContentConflict) as exc:
        ingest_file(c, conn, cfg, target_doc_id=ra.doc_id)

    assert exc.value.doc_id == rb.doc_id
    assert str(exc.value) == f"this content is already stored as document {rb.doc_id}"
    assert snapshot() == before


def test_explicit_update_unknown_doc_refused(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)

    with pytest.raises(DocumentNotFound) as exc:
        ingest_file(a, conn, cfg, target_doc_id="nope")

    assert exc.value.doc_id == "nope"
    assert str(exc.value) == "document nope not found"
    assert _count(conn, "SELECT COUNT(*) FROM documents") == 0
    assert _count(conn, "SELECT COUNT(*) FROM document_sources") == 0


def test_explicit_update_source_owned_by_other_doc_refused(
    cfg, conn, tmp_path, fake_embed
):
    a = _write(tmp_path / "a.md", ORIGINAL)
    b = _write(tmp_path / "b.md", "# B\n\ngamma owls\n")
    ra = ingest_file(a, conn, cfg)
    rb = ingest_file(b, conn, cfg)

    def snapshot():
        return (
            _doc(conn, ra.doc_id), _doc(conn, rb.doc_id),
            _sources(conn, ra.doc_id), _sources(conn, rb.doc_id),
            _count(conn, "SELECT COUNT(*) FROM documents"),
            _count(conn, "SELECT COUNT(*) FROM chunks"),
        )

    before = snapshot()
    _append(b, "\nbeta ravens\n")
    with pytest.raises(SourceConflict) as exc:
        ingest_file(b, conn, cfg, target_doc_id=ra.doc_id)

    assert exc.value.doc_id == rb.doc_id
    assert str(exc.value) == f"this source already belongs to document {rb.doc_id}"
    assert snapshot() == before


def test_explicit_upload_filename_shared_with_other_doc_is_not_a_conflict(
    cfg, conn, tmp_path, fake_embed
):
    ingest_fetched(Fetched(b"first notes", "upload", "upload:notes.md"), conn, cfg)
    b = _write(tmp_path / "b.md", "# B\n\ngamma owls\n")
    rb = ingest_file(b, conn, cfg)

    r = ingest_fetched(
        Fetched(b"second notes", "upload", "upload:notes.md"), conn, cfg,
        target_doc_id=rb.doc_id,
    )

    assert r.was_updated is True
    assert r.doc_id == rb.doc_id


def test_explicit_update_can_change_format(cfg, conn, tmp_path, fake_embed):
    a = _write(tmp_path / "a.md", ORIGINAL)
    d = _write(tmp_path / "d.txt", "plain delta text\n")
    ra = ingest_file(a, conn, cfg)

    r = ingest_file(d, conn, cfg, target_doc_id=ra.doc_id)

    assert r.content_type == "txt"
    sha, ctype, _ = lookup_blob_for(conn, ra.doc_id)
    assert (sha, ctype) == (sha256_bytes(d.read_bytes()), "txt")
