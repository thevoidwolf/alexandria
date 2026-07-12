"""Nearest-neighbor tag/category suggestion — no LLM required.

Reuses the chunk embeddings that already power search. For a target doc:

1. Average the doc's chunk embeddings into a doc-level vector.
2. k-NN in ``chunks_vec`` for the top-K nearby chunks, drop the doc's own.
3. Aggregate per neighbor doc (best-similarity chunk wins per doc).
4. Vote: any category on ≥ ``category_min_fraction`` of neighbors becomes
   the suggested category; any tag on ≥ ``tag_min_fraction`` of neighbors
   becomes a suggested tag. Confidence is the agreement fraction.

Deterministic, inspectable (``neighbors`` field shows the docs the
suggestion was drawn from), and gets better automatically as the user
curates — no retraining loop, no external model, no network.
"""
from __future__ import annotations

import sqlite3
import struct
from collections import Counter
from dataclasses import dataclass

import numpy as np

from alexandria.catalog import display_title
from alexandria.config import Config

# Chunk-level over-fetch. Personal corpora rarely need more; large ones can
# bump this if per-doc dedupe eats too much of the pool.
_K_CHUNKS = 60
# Cap on unique neighbor docs used for voting.
_MAX_NEIGHBORS = 15


@dataclass(frozen=True)
class Neighbor:
    doc_id: str
    display_title: str
    similarity: float           # (0, 1]; larger = closer
    category: str | None
    tags: list[str]


@dataclass(frozen=True)
class Suggestion:
    doc_id: str
    current: dict                       # {title, category, tags}
    suggested_category: str | None
    suggested_tags: list[str]
    category_confidence: float          # 0.0 when no category met threshold
    tag_confidences: dict[str, float]   # only for suggested_tags
    neighbors: list[Neighbor]
    applied: bool = False


def _serialize_vec(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def _distance_to_similarity(distance: float) -> float:
    """Monotonic map to (0, 1]. Exact scale doesn't matter for voting —
    this just gives the API a friendlier number than a raw distance."""
    return 1.0 / (1.0 + max(0.0, distance))


def _load_doc_mean_embedding(
    conn: sqlite3.Connection, doc_id: str
) -> tuple[list[int], np.ndarray] | None:
    """Return (own_chunk_rowids, unit-normalized mean vector) or None.

    Joining chunks_vec by chunk_rowid gives us the packed float32 bytes;
    ``np.frombuffer`` decodes without a copy.
    """
    rows = conn.execute(
        """
        SELECT c.rowid, v.embedding
        FROM chunks c
        JOIN chunks_vec v ON v.chunk_rowid = c.rowid
        WHERE c.doc_id = ?
        """,
        (doc_id,),
    ).fetchall()
    if not rows:
        return None
    rowids = [r[0] for r in rows]
    vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    mean = vecs.mean(axis=0)
    # Re-normalize so the mean vector is comparable to the unit-normalized
    # chunk embeddings sqlite-vec is comparing against.
    norm = float(np.linalg.norm(mean))
    if norm > 0:
        mean = mean / norm
    return rowids, mean.astype(np.float32)


def _knn_chunks(
    conn: sqlite3.Connection, query_vec_bytes: bytes, k: int
) -> list[tuple[int, float]]:
    return conn.execute(
        "SELECT chunk_rowid, distance FROM chunks_vec "
        "WHERE embedding MATCH ? AND k = ?",
        (query_vec_bytes, k),
    ).fetchall()


def _best_per_doc(
    conn: sqlite3.Connection,
    hits: list[tuple[int, float]],
    exclude_rowids: set[int],
    limit: int,
) -> list[tuple[str, float]]:
    """Group chunk hits by doc, keep smallest-distance per doc, drop self."""
    filtered = [(r, d) for r, d in hits if r not in exclude_rowids]
    if not filtered:
        return []
    placeholders = ",".join("?" * len(filtered))
    rowid_to_doc: dict[int, str] = dict(conn.execute(
        f"SELECT rowid, doc_id FROM chunks WHERE rowid IN ({placeholders})",
        [r for r, _ in filtered],
    ).fetchall())

    best: dict[str, float] = {}
    for rowid, dist in filtered:
        doc = rowid_to_doc.get(rowid)
        if not doc:
            continue
        if doc not in best or dist < best[doc]:
            best[doc] = dist
    # Ascending distance = descending similarity.
    return sorted(best.items(), key=lambda kv: kv[1])[:limit]


def _load_neighbor_meta(
    conn: sqlite3.Connection, doc_ids: list[str]
) -> dict[str, tuple[str | None, str | None, list[str], str, str | None]]:
    """doc_id → (title, category, tags, content_type, first source_uri)."""
    if not doc_ids:
        return {}
    placeholders = ",".join("?" * len(doc_ids))
    docs = conn.execute(
        f"SELECT id, title, category, content_type FROM documents "
        f"WHERE id IN ({placeholders})",
        doc_ids,
    ).fetchall()
    tags: dict[str, list[str]] = {}
    for row in conn.execute(
        f"SELECT doc_id, tag FROM tags WHERE doc_id IN ({placeholders}) ORDER BY tag",
        doc_ids,
    ):
        tags.setdefault(row[0], []).append(row[1])
    sources: dict[str, str] = {}
    for row in conn.execute(
        f"SELECT doc_id, source_uri FROM document_sources "
        f"WHERE doc_id IN ({placeholders}) ORDER BY seen_at",
        doc_ids,
    ):
        sources.setdefault(row[0], row[1])  # first-seen wins

    return {
        doc_id: (title, category, tags.get(doc_id, []), ctype, sources.get(doc_id))
        for doc_id, title, category, ctype in docs
    }


def _empty_suggestion(
    doc_id: str,
    current_title: str | None,
    current_category: str | None,
    current_tags: list[str],
) -> Suggestion:
    return Suggestion(
        doc_id=doc_id,
        current={"title": current_title, "category": current_category, "tags": current_tags},
        suggested_category=None,
        suggested_tags=[],
        category_confidence=0.0,
        tag_confidences={},
        neighbors=[],
    )


def suggest_metadata(
    doc_id: str, conn: sqlite3.Connection, cfg: Config
) -> Suggestion | None:
    """Suggest a category + tags for ``doc_id`` from its embedding neighbors.

    Returns None when the doc is unknown. Returns a Suggestion with empty
    fields when the corpus has no labeled neighbors yet — callers can tell
    "unknown doc" (None) from "no signal" (empty suggestion).
    """
    row = conn.execute(
        "SELECT title, category, content_type FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        return None
    current_title, current_category, _ctype = row
    current_tags = [t[0] for t in conn.execute(
        "SELECT tag FROM tags WHERE doc_id = ? ORDER BY tag", (doc_id,)
    )]

    loaded = _load_doc_mean_embedding(conn, doc_id)
    if loaded is None:
        return _empty_suggestion(doc_id, current_title, current_category, current_tags)
    own_rowids, mean_vec = loaded

    hits = _knn_chunks(conn, _serialize_vec(mean_vec), _K_CHUNKS)
    neighbor_pairs = _best_per_doc(conn, hits, set(own_rowids), _MAX_NEIGHBORS)
    if not neighbor_pairs:
        return _empty_suggestion(doc_id, current_title, current_category, current_tags)

    meta = _load_neighbor_meta(conn, [d for d, _ in neighbor_pairs])

    neighbors: list[Neighbor] = []
    cat_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    for doc_id_n, dist in neighbor_pairs:
        m = meta.get(doc_id_n)
        if not m:
            continue
        title_n, category_n, tags_n, ctype_n, source_uri = m
        neighbors.append(Neighbor(
            doc_id=doc_id_n,
            display_title=display_title(title_n, source_uri, ctype_n),
            similarity=_distance_to_similarity(dist),
            category=category_n,
            tags=sorted(tags_n),
        ))
        if category_n:
            cat_counts[category_n] += 1
        for tag in tags_n:
            tag_counts[tag] += 1

    n_total = len(neighbors)
    if n_total == 0:
        return _empty_suggestion(doc_id, current_title, current_category, current_tags)

    cat_min = cfg.classify.category_min_fraction
    tag_min = cfg.classify.tag_min_fraction
    max_tags = cfg.classify.max_tags

    suggested_category: str | None = None
    category_confidence = 0.0
    if cat_counts:
        top_cat, top_n = cat_counts.most_common(1)[0]
        frac = top_n / n_total
        if frac >= cat_min:
            suggested_category = top_cat
            category_confidence = frac

    tag_fracs = {t: c / n_total for t, c in tag_counts.items() if c / n_total >= tag_min}
    suggested_tags = sorted(tag_fracs, key=tag_fracs.__getitem__, reverse=True)[:max_tags]

    return Suggestion(
        doc_id=doc_id,
        current={"title": current_title, "category": current_category, "tags": current_tags},
        suggested_category=suggested_category,
        suggested_tags=suggested_tags,
        category_confidence=category_confidence,
        tag_confidences={t: tag_fracs[t] for t in suggested_tags},
        neighbors=neighbors,
    )
