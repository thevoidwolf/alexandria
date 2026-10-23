"""Eval datasets: a corpus, queries and graded relevance judgements."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dataset:
    name: str
    docs: dict[str, str]                    # dataset doc id -> document body
    queries: dict[str, str]                 # query id -> query text
    qrels: dict[str, dict[str, int]]        # query id -> {doc id: grade}
    source: dict[str, str] = field(default_factory=dict)  # provenance, e.g. pinned revisions
