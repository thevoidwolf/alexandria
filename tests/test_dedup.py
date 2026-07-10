from pathlib import Path

from alexandria.ingest import ingest_file


def test_source_gate_hits_on_same_uri(cfg, conn, corpus_dir: Path, fake_embed):
    r1 = ingest_file(corpus_dir / "plain.txt", conn, cfg, category="bills")
    r2 = ingest_file(corpus_dir / "plain.txt", conn, cfg, category="bills")

    assert r1.was_duplicate is False
    assert r2.was_duplicate is True
    assert r2.dedup_gate == "source"
    assert r2.doc_id == r1.doc_id


def test_raw_gate_hits_on_same_bytes_different_path(
    cfg, conn, corpus_dir: Path, fake_embed
):
    src = corpus_dir / "plain.txt"
    dup = corpus_dir / "plain-copy.txt"
    dup.write_bytes(src.read_bytes())

    r1 = ingest_file(src, conn, cfg)
    r2 = ingest_file(dup, conn, cfg)

    assert r2.was_duplicate is True
    assert r2.dedup_gate == "raw"
    assert r2.doc_id == r1.doc_id


def test_text_gate_hits_when_bytes_differ_but_normalized_text_matches(
    cfg, conn, corpus_dir: Path, fake_embed
):
    src = corpus_dir / "plain.txt"
    variant = corpus_dir / "plain-newlines.txt"
    variant.write_bytes(src.read_bytes() + b"\n\n\n")

    r1 = ingest_file(src, conn, cfg)
    r2 = ingest_file(variant, conn, cfg)

    assert r2.was_duplicate is True
    assert r2.dedup_gate == "text"
    assert r2.doc_id == r1.doc_id


def test_duplicate_registers_alternate_source(
    cfg, conn, corpus_dir: Path, fake_embed
):
    src = corpus_dir / "plain.txt"
    dup = corpus_dir / "plain-copy.txt"
    dup.write_bytes(src.read_bytes())

    ingest_file(src, conn, cfg)
    ingest_file(dup, conn, cfg)

    rows = conn.execute(
        "SELECT source_uri FROM document_sources ORDER BY source_uri"
    ).fetchall()
    uris = [r[0] for r in rows]
    assert str(src) in uris
    assert str(dup) in uris


def test_duplicate_merges_new_tags(cfg, conn, corpus_dir: Path, fake_embed):
    r1 = ingest_file(corpus_dir / "plain.txt", conn, cfg, tags=["utility"])
    ingest_file(corpus_dir / "plain.txt", conn, cfg, tags=["electric"])
    tags = [r[0] for r in conn.execute(
        "SELECT tag FROM tags WHERE doc_id = ? ORDER BY tag", (r1.doc_id,)
    )]
    assert tags == ["electric", "utility"]
