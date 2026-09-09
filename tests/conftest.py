from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from alexandria.config import Config, load as load_config
from alexandria.db import connect

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """Isolated ALEXANDRIA_HOME per test."""
    h = tmp_path / "alexandria"
    h.mkdir()
    return h


@pytest.fixture
def cfg(home: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("ALEXANDRIA_HOME", str(home))
    return load_config()


@pytest.fixture
def conn(cfg: Config):
    c = connect(cfg.db_path, cfg.embeddings.dim)
    yield c
    c.close()


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Fresh copy of the fixture corpus under tmp_path so tests can mutate it."""
    dst = tmp_path / "corpus"
    shutil.copytree(FIXTURES, dst)
    return dst


@pytest.fixture
def fake_embed(monkeypatch: pytest.MonkeyPatch):
    """Replace the sentence-transformer embedder with a deterministic hash-based one.

    Use in tests that exercise ingest/dedup/catalog paths without caring about
    retrieval quality. Vectors are stable per input text, so cosine search still
    works; they're just not semantically meaningful.
    """
    import hashlib

    import numpy as np

    def _fake(texts, cfg):
        dim = cfg.dim
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            out[i] = v
        return out

    monkeypatch.setattr("alexandria.ingest.orchestrator.embed_texts", _fake)
    monkeypatch.setattr("alexandria.search.embed_texts", _fake)
