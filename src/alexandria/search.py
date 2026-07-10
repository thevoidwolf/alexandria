from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from typing import Literal

from alexandria.config import Config
from alexandria.ingest.embed import embed_texts

SearchMode = Literal["hybrid", "fts", "vec"]

# RRF constant. Anserini's default; robust across corpora, no tuning knob.
RRF_K = 60


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    score: float
    snippet: str
    title: str | None
    category: str | None
    tags: list[str]
    source_uri: str | None
    content_type: str


def _serialize_vec(vec) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def _eligible_chunk_rowids(
    conn: sqlite3.Connection,
    category: str | None,
    tags: list[str] | None,
) -> list[int] | None:
    """Return the set of chunk rowids allowed by the filters, or None to mean 'all'."""
    if not category and not tags:
        return None

    clauses: list[str] = []
    params: list = []
    sql = "SELECT c.rowid FROM chunks c JOIN documents d ON d.id = c.doc_id"

    if tags:
        # AND semantics: doc must carry every requested tag.
        placeholders = ",".join("?" * len(tags))
        sql += (
            f" JOIN (SELECT doc_id FROM tags WHERE tag IN ({placeholders}) "
            f"GROUP BY doc_id HAVING COUNT(DISTINCT tag) = ?) t ON t.doc_id = c.doc_id"
        )
        params.extend(tags)
        params.append(len(tags))

    if category:
        clauses.append("d.category = ?")
        params.append(category)

    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    rows = conn.execute(sql, params).fetchall()
    return [r[0] for r in rows]


def _fts_search(
    conn: sqlite3.Connection,
    query: str,
    eligible: list[int] | None,
    limit: int,
) -> list[tuple[int, str, str, str]]:
    """Return [(rowid, chunk_id, doc_id, snippet)] ordered by BM25 rank."""
    if eligible is not None and not eligible:
        return []

    sql = (
        "SELECT rowid, chunk_id, doc_id, "
        "snippet(chunks_fts, 2, '[', ']', '...', 12) "
        "FROM chunks_fts WHERE chunks_fts MATCH ?"
    )
    params: list = [query]

    if eligible is not None:
        placeholders = ",".join("?" * len(eligible))
        sql += f" AND rowid IN ({placeholders})"
        params.extend(eligible)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        # FTS5 rejects some query strings (e.g. bare punctuation). Treat as no matches.
        return []


def _vec_search(
    conn: sqlite3.Connection,
    query_vec_bytes: bytes,
    eligible: list[int] | None,
    limit: int,
) -> list[tuple[int, float]]:
    """Return [(rowid, distance)] from sqlite-vec KNN."""
    if eligible is not None and not eligible:
        return []

    sql = (
        "SELECT chunk_rowid, distance FROM chunks_vec "
        "WHERE embedding MATCH ? AND k = ?"
    )
    params: list = [query_vec_bytes, limit]

    if eligible is not None:
        placeholders = ",".join("?" * len(eligible))
        sql += f" AND chunk_rowid IN ({placeholders})"
        params.extend(eligible)

    return conn.execute(sql, params).fetchall()


def _load_hit(conn: sqlite3.Connection, rowid: int) -> tuple[str, str, str, str, str | None, str | None, str]:
    """Return (chunk_id, doc_id, chunk_text, title, category, source_uri, content_type)."""
    row = conn.execute(
        """
        SELECT
            c.id,
            c.doc_id,
            c.text,
            d.title,
            d.category,
            (SELECT source_uri FROM document_sources s
              WHERE s.doc_id = d.id ORDER BY s.seen_at LIMIT 1),
            d.content_type
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        WHERE c.rowid = ?
        """,
        (rowid,),
    ).fetchone()
    return row


def _load_tags(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    return [
        r[0] for r in conn.execute(
            "SELECT tag FROM tags WHERE doc_id = ? ORDER BY tag", (doc_id,)
        )
    ]


def _rrf(rankings: list[list[int]]) -> dict[int, float]:
    """Reciprocal rank fusion. Input: list of ranked rowid lists. Output: rowid → fused score."""
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, rowid in enumerate(ranked, start=1):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def search(
    query: str,
    conn: sqlite3.Connection,
    cfg: Config,
    category: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
    mode: SearchMode = "hybrid",
) -> list[SearchHit]:
    """Hybrid keyword + semantic retrieval, RRF-fused, category/tag filtered."""
    if not query.strip():
        return []

    eligible = _eligible_chunk_rowids(conn, category, tags)
    over_fetch = max(limit * 3, 30)

    fts_rows = _fts_search(conn, query, eligible, over_fetch) if mode in ("hybrid", "fts") else []
    fts_by_rowid: dict[int, str] = {r[0]: r[3] for r in fts_rows}  # rowid → snippet
    fts_ranked = [r[0] for r in fts_rows]

    vec_ranked: list[int] = []
    if mode in ("hybrid", "vec"):
        q_vec = embed_texts([query], cfg.embeddings)[0]
        vec_rows = _vec_search(conn, _serialize_vec(q_vec), eligible, over_fetch)
        vec_ranked = [r[0] for r in vec_rows]

    if mode == "fts":
        fused = {rowid: 1.0 / (RRF_K + i + 1) for i, rowid in enumerate(fts_ranked)}
    elif mode == "vec":
        fused = {rowid: 1.0 / (RRF_K + i + 1) for i, rowid in enumerate(vec_ranked)}
    else:
        fused = _rrf([fts_ranked, vec_ranked])

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]

    hits: list[SearchHit] = []
    for rowid, score in ordered:
        row = _load_hit(conn, rowid)
        if row is None:
            continue
        chunk_id, doc_id, chunk_text, title, doc_category, source_uri, content_type = row
        snippet = fts_by_rowid.get(rowid)
        if not snippet:
            snippet = chunk_text[:200] + ("…" if len(chunk_text) > 200 else "")
        hits.append(
            SearchHit(
                chunk_id=chunk_id,
                doc_id=doc_id,
                score=score,
                snippet=snippet,
                title=title,
                category=doc_category,
                tags=_load_tags(conn, doc_id),
                source_uri=source_uri,
                content_type=content_type,
            )
        )
    return hits
