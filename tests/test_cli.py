from pathlib import Path

from typer.testing import CliRunner

from alexandria.cli import app

runner = CliRunner()


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def test_cli_ingest_reports_update(cfg, conn, tmp_path: Path, fake_embed):
    a = _write(tmp_path / "a.md", "# A\n\nfirst version text\n")
    assert runner.invoke(app, ["ingest", str(a)]).exit_code == 0
    a.write_text("# A\n\nsecond version with different text\n")
    res = runner.invoke(app, ["ingest", str(a)])
    assert res.exit_code == 0
    assert "updated" in res.output


def test_cli_replaces_updates_target_from_new_path(
    cfg, conn, tmp_path: Path, fake_embed
):
    a = _write(tmp_path / "a.md", "# A\n\nfirst version text\n")
    b = _write(tmp_path / "b.md", "# B\n\ncompletely different words here\n")
    runner.invoke(app, ["ingest", str(a)])
    doc_id = conn.execute("SELECT id FROM documents").fetchone()[0]
    res = runner.invoke(app, ["ingest", str(b), "--replaces", doc_id])
    assert res.exit_code == 0
    assert "updated" in res.output
    rows = conn.execute("SELECT id FROM documents").fetchall()
    assert len(rows) == 1
    text = " ".join(
        r[0] for r in conn.execute("SELECT text FROM chunks WHERE doc_id = ?", (doc_id,))
    )
    assert "completely different words" in text


def test_cli_replaces_unknown_doc_exits_1(cfg, conn, tmp_path: Path, fake_embed):
    a = _write(tmp_path / "a.md", "# A\n\nsome text\n")
    res = runner.invoke(app, ["ingest", str(a), "--replaces", "nope"])
    assert res.exit_code == 1
    assert "not found" in res.output


def test_cli_ingest_folder_counts_updates(cfg, conn, corpus_dir: Path, fake_embed):
    assert runner.invoke(app, ["ingest-folder", str(corpus_dir)]).exit_code == 0
    note = corpus_dir / "note.md"
    note.write_text(note.read_text() + "\n\nAn added paragraph of new content.\n")
    res = runner.invoke(app, ["ingest-folder", str(corpus_dir)])
    assert res.exit_code == 0
    assert "1 updated" in res.output
