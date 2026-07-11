"""Ensure a v1 DB is picked up as v2 without data loss."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from alexandria.db import SCHEMA_VERSION, connect


def _touch_v1_db(path: Path, dim: int) -> None:
    """Simulate a pre-v2 DB: v1 schema + version=1 stamp."""
    conn = connect(path, dim)
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version(version) VALUES (1)")
    conn.execute("DROP TABLE jobs")
    conn.close()


def test_v1_db_migrates_to_v2(tmp_path: Path):
    db = tmp_path / "alexandria.db"
    _touch_v1_db(db, dim=8)

    with sqlite3.connect(db) as raw:
        assert raw.execute("SELECT version FROM schema_version").fetchone()[0] == 1
        tables = {r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "jobs" not in tables

    conn = connect(db, embed_dim=8)
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION

    conn.execute("INSERT INTO jobs(id, kind, status, created_at, updated_at, input) "
                 "VALUES ('j1','upload','queued','t','t','{}')")
    row = conn.execute("SELECT id, kind, status FROM jobs WHERE id = 'j1'").fetchone()
    assert row == ("j1", "upload", "queued")
    conn.close()


def test_future_schema_version_raises(tmp_path: Path):
    import pytest

    db = tmp_path / "alexandria.db"
    conn = connect(db, embed_dim=8)
    conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION + 99,))
    conn.close()

    with pytest.raises(RuntimeError, match="newer than code version"):
        connect(db, embed_dim=8)
