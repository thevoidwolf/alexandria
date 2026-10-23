"""Eval datasets: a corpus, queries and graded relevance judgements."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dataset:
    name: str
    # dataset doc id -> body. Text is stored the way store_document_tool
    # stores it; bytes (e.g. a PDF) the way a web upload is.
    docs: dict[str, str | bytes]
    queries: dict[str, str]                 # query id -> query text
    qrels: dict[str, dict[str, int]]        # query id -> {doc id: grade}
    source: dict[str, str] = field(default_factory=dict)  # provenance, e.g. pinned revisions
    doc_suffix: str = ".txt"                # filename extension each doc is ingested under
    titles: dict[str, str] = field(default_factory=dict)  # doc id -> human-readable title
    query_meta: dict[str, dict] = field(default_factory=dict)  # query id -> {type, author}
