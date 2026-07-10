from __future__ import annotations

import fnmatch
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from alexandria.config import Config
from alexandria.ingest.orchestrator import IngestResult, ingest_file

# Default extension allow-list for `ingest_folder` when no --glob is given.
SUPPORTED_EXTS = {".pdf", ".html", ".htm", ".txt", ".md", ".markdown"}


@dataclass
class WalkResult:
    ingested: list[str] = field(default_factory=list)   # doc_ids of new inserts
    duplicates: list[str] = field(default_factory=list) # doc_ids that hit a dedup gate
    errors: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)

    @property
    def scanned(self) -> int:
        return len(self.ingested) + len(self.duplicates) + len(self.errors)


def _iter_paths(root: Path, recursive: bool, glob: str | None) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    paths: list[Path] = []
    walker = root.rglob("*") if recursive else root.iterdir()
    for p in walker:
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue  # skip dotdirs / dotfiles
        if glob:
            if not fnmatch.fnmatch(p.name, glob):
                continue
        else:
            if p.suffix.lower() not in SUPPORTED_EXTS:
                continue
        paths.append(p)
    paths.sort()
    return paths


def ingest_folder(
    root: Path,
    conn: sqlite3.Connection,
    cfg: Config,
    recursive: bool = True,
    glob: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
) -> WalkResult:
    result = WalkResult()
    for path in _iter_paths(root, recursive=recursive, glob=glob):
        try:
            r: IngestResult = ingest_file(path, conn, cfg, category=category, tags=tags)
        except Exception as exc:
            result.errors.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue
        if r.was_duplicate:
            result.duplicates.append(r.doc_id)
        else:
            result.ingested.append(r.doc_id)
    return result
