from pathlib import Path

from alexandria.ingest import ingest_folder


def test_walker_ingests_all_supported_types(cfg, conn, corpus_dir: Path, fake_embed):
    result = ingest_folder(corpus_dir, conn, cfg, category="mixed")
    assert len(result.ingested) == 6   # plain, note, page, paper.md, paper.pdf, second-bill
    assert result.errors == []


def test_walker_second_run_is_all_duplicates(
    cfg, conn, corpus_dir: Path, fake_embed
):
    ingest_folder(corpus_dir, conn, cfg)
    result = ingest_folder(corpus_dir, conn, cfg)
    assert result.ingested == []
    assert len(result.duplicates) == 6


def test_walker_glob_filters_by_name(cfg, conn, corpus_dir: Path, fake_embed):
    result = ingest_folder(corpus_dir, conn, cfg, glob="*.md")
    # only note.md and paper.md
    assert len(result.ingested) == 2
    docs = conn.execute("SELECT content_type FROM documents").fetchall()
    assert all(row[0] == "md" for row in docs)


def test_walker_skips_dotfiles(cfg, conn, corpus_dir: Path, fake_embed):
    (corpus_dir / ".hidden.md").write_text("secret")
    result = ingest_folder(corpus_dir, conn, cfg)
    ingested_paths = [
        r[0] for r in conn.execute(
            "SELECT source_uri FROM document_sources"
        )
    ]
    assert not any(".hidden.md" in p for p in ingested_paths)
    assert len(result.ingested) == 6


def test_walker_records_error_and_continues(
    cfg, conn, corpus_dir: Path, fake_embed
):
    # An empty text file has no extractable content — orchestrator raises ValueError.
    (corpus_dir / "empty.md").write_text("")
    result = ingest_folder(corpus_dir, conn, cfg)
    assert len(result.errors) == 1
    assert result.errors[0][0].endswith("empty.md")
    assert len(result.ingested) == 6  # other files still ingested


def test_walker_no_recursion_ignores_subdirs(
    cfg, conn, corpus_dir: Path, fake_embed
):
    sub = corpus_dir / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("deep content unique text")
    ingest_folder(corpus_dir, conn, cfg, recursive=True)
    assert any("deep.txt" in u for u in [
        row[0] for row in conn.execute("SELECT source_uri FROM document_sources")
    ])

    # reset by opening a fresh DB dir for the second scenario
    # (or just check the shallow walker wouldn't have found deep.txt on its own)
    from alexandria.ingest.walker import _iter_paths
    shallow = _iter_paths(corpus_dir, recursive=False, glob=None)
    assert not any(p.name == "deep.txt" for p in shallow)
