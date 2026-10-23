"""Build and cache an isolated Alexandria home per (corpus, index settings).

Each index lives in ``<cache_root>/<key>/home`` and is opened with an
explicit home, so the user's own corpus (``ALEXANDRIA_HOME`` or the XDG
default) is never read or written. Documents go in through
``ingest_fetched``: text exactly as ``store_document_tool`` sends it, bytes
(PDFs) exactly as a web upload does, so what is measured is the real
extract/normalize/chunk/embed pipeline.

The cache key covers the corpus, the settings, the embedding model's Hub
revision and the source of the ingest code, so an ingest fix or a moved
model rebuilds the index instead of reusing a stale one.
``build.json`` is written last; a directory without it is a crashed build
and is thrown away.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import alexandria.db
import alexandria.ingest
from alexandria.config import Config, EmbeddingsConfig, load as load_config
from alexandria.db import connect
from alexandria.ingest.embed import _get_model
from alexandria.ingest.fetch import Fetched
from alexandria.ingest.orchestrator import ingest_fetched
from evals.datasets import Dataset


@dataclass(frozen=True)
class IndexSettings:
    """Everything fixed at ingest time: changing any of it needs a new index."""

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    chunk_tokens: int = 800
    chunk_overlap: int = 100
    pdf_backend: str = "pypdf"


@dataclass
class BuiltIndex:
    home: Path
    doc_map: dict[str, str]   # dataset doc id -> Alexandria doc id
    stats: dict


def _code_fingerprint() -> bytes:
    h = hashlib.sha256()
    files = sorted(Path(alexandria.ingest.__file__).parent.glob("*.py"))
    for f in [*files, Path(alexandria.db.__file__)]:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.digest()


def model_revision(name: str) -> str:
    """Hub commit of the embedding model Alexandria loads for ``name``.

    Alexandria loads the model's ``main`` revision and has no setting to pin
    one, so the eval records which commit that was. The model is loaded
    first, exactly as ingest would (downloading it on a fresh machine, or
    picking up a moved ``main``), then the revision is read from the cache.
    """
    from huggingface_hub import try_to_load_from_cache

    _get_model(name, EmbeddingsConfig().device)
    path = try_to_load_from_cache(name, "config.json")
    return Path(path).parent.name if isinstance(path, str) else "not-from-hub"


def index_key(ds: Dataset, s: IndexSettings) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(
        {"dataset": ds.name, "embed_revision": model_revision(s.embed_model), **asdict(s)},
        sort_keys=True,
    ).encode())
    h.update(ds.doc_suffix.encode() + b"\0")
    for doc_id in sorted(ds.docs):
        body = ds.docs[doc_id]
        h.update(doc_id.encode() + b"\0" + (body if isinstance(body, bytes) else body.encode()) + b"\0")
    h.update(_code_fingerprint())
    return f"{ds.name}-{h.hexdigest()[:16]}"


def _write_config(home: Path, s: IndexSettings) -> None:
    (home / "config.toml").write_text(
        "[embeddings]\n"
        f"model = {json.dumps(s.embed_model)}\n"
        f"dim = {s.embed_dim}\n\n"
        "[chunking]\n"
        f"tokens = {s.chunk_tokens}\n"
        f"overlap = {s.chunk_overlap}\n\n"
        "[storage]\n"
        "keep_blobs = false\n\n"
        "[extractors.pdf]\n"
        f"backend = {json.dumps(s.pdf_backend)}\n"
    )


def build_index(ds: Dataset, s: IndexSettings, cache_root: Path) -> BuiltIndex:
    root = cache_root / index_key(ds, s)
    home = root / "home"
    marker = root / "build.json"
    if marker.exists():
        return BuiltIndex(
            home=home,
            doc_map=json.loads((root / "docmap.json").read_text()),
            stats=json.loads(marker.read_text()),
        )
    if root.exists():
        shutil.rmtree(root)
    home.mkdir(parents=True)
    _write_config(home, s)
    cfg = load_config(home)
    conn = connect(cfg.db_path, cfg.embeddings.dim)

    doc_map: dict[str, str] = {}
    failed: dict[str, str] = {}
    n_collapsed = 0
    start = time.perf_counter()
    try:
        for ds_id in sorted(ds.docs):
            body = ds.docs[ds_id]
            name = f"{ds_id}{ds.doc_suffix}"
            if isinstance(body, bytes):
                fetched = Fetched(body, "upload", f"upload:{name}")
            else:
                fetched = Fetched(body.encode("utf-8"), "mcp", f"mcp:{name}")
            try:
                r = ingest_fetched(fetched, conn, cfg, filename_hint=name)
            except ValueError as e:  # e.g. no extractable text
                failed[ds_id] = str(e)
                continue
            doc_map[ds_id] = r.doc_id
            n_collapsed += r.was_duplicate
        build_seconds = time.perf_counter() - start
        (n_docs,) = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        (n_chunks,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()

    stats = {
        "key": root.name,
        "settings": asdict(s),
        "build_seconds": round(build_seconds, 2),
        "db_bytes": cfg.db_path.stat().st_size,
        "n_docs": n_docs,
        "n_chunks": n_chunks,
        "n_collapsed": n_collapsed,
        "failed": failed,
        "embed_device": cfg.embeddings.device,
        "embed_revision": model_revision(s.embed_model),
        "python": platform.python_version(),
    }
    (root / "docmap.json").write_text(json.dumps(doc_map, sort_keys=True))
    marker.write_text(json.dumps(stats, indent=2, sort_keys=True))
    return BuiltIndex(home=home, doc_map=doc_map, stats=stats)


def open_index(b: BuiltIndex, **embedding_overrides) -> tuple[Config, sqlite3.Connection]:
    """Config and connection for a built index; overrides apply to queries only."""
    cfg = load_config(b.home)
    if embedding_overrides:
        cfg = replace(cfg, embeddings=replace(cfg.embeddings, **embedding_overrides))
    return cfg, connect(cfg.db_path, cfg.embeddings.dim)


def canonical_ids(doc_map: dict[str, str]) -> dict[str, str]:
    """Alexandria doc id -> the first dataset id (in ingest order) stored as it.

    Alexandria ids are time-based ULIDs that change on every build; results
    are recorded in dataset ids so a rebuild reproduces them exactly.
    """
    out: dict[str, str] = {}
    for ds_id in sorted(doc_map):
        out.setdefault(doc_map[ds_id], ds_id)
    return out


def translate_qrels(
    qrels: dict[str, dict[str, int]], doc_map: dict[str, str]
) -> dict[str, dict[str, int]]:
    """Qrels re-keyed by canonical id: identical docs that collapsed into one
    count once, at their highest grade. Judged docs that never made it into
    the index keep their id, so they still count as relevant-but-missed."""
    canon = canonical_ids(doc_map)
    out: dict[str, dict[str, int]] = {}
    for qid, judged in qrels.items():
        merged: dict[str, int] = {}
        for d, grade in judged.items():
            c = canon[doc_map[d]] if d in doc_map else d
            merged[c] = max(grade, merged.get(c, grade))
        out[qid] = merged
    return out
