"""Label anchors — user-authored prototype descriptions for tags/categories.

An anchor is a (kind, name, description) triple where ``description`` is a
short natural-language definition of what the label means. The description
is embedded once at write time using the same model that embeds document
chunks; classification later computes cosine similarity between a doc's
mean chunk vector and each anchor's vector.

Anchors let the classifier work well *before* the corpus has enough
correctly-labeled documents to bootstrap from. They're independent of the
existing (possibly noisy) taxonomy: an anchor's confidence comes from the
user's description, not from any other document's labels.
"""
from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from alexandria.config import Config
from alexandria.ingest.embed import embed_texts


ALLOWED_KINDS = ("category", "tag")


@dataclass(frozen=True)
class Anchor:
    kind: str
    name: str
    description: str
    embed_model: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AnchorMatch:
    """One anchor's similarity to a target vector."""
    kind: str
    name: str
    similarity: float           # cosine, [-1, 1]; for unit vectors we clamp to (0, 1]
    description: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _serialize_vec(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def _validate_kind(kind: str) -> None:
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"kind must be one of {ALLOWED_KINDS}, got {kind!r}")


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("anchor name may not be blank")
    return name


def _row_to_anchor(row) -> Anchor:
    return Anchor(
        kind=row[0], name=row[1], description=row[2],
        embed_model=row[3], created_at=row[4], updated_at=row[5],
    )


# ---- CRUD -----------------------------------------------------------------


def list_anchors(conn: sqlite3.Connection) -> list[Anchor]:
    rows = conn.execute(
        "SELECT kind, name, description, embed_model, created_at, updated_at "
        "FROM label_anchors ORDER BY kind, name"
    ).fetchall()
    return [_row_to_anchor(r) for r in rows]


def get_anchor(
    conn: sqlite3.Connection, kind: str, name: str
) -> Anchor | None:
    row = conn.execute(
        "SELECT kind, name, description, embed_model, created_at, updated_at "
        "FROM label_anchors WHERE kind = ? AND name = ?",
        (kind, name),
    ).fetchone()
    return _row_to_anchor(row) if row else None


def set_anchor(
    conn: sqlite3.Connection,
    cfg: Config,
    kind: str,
    name: str,
    description: str,
) -> Anchor:
    """Upsert an anchor. Re-embeds on every write.

    Even if only ``description`` changes we re-embed unconditionally —
    the cost is a single sentence-transformer forward pass; keeping the
    embedding in sync with the description is worth the CPU.
    """
    _validate_kind(kind)
    name = _validate_name(name)
    if not (description or "").strip():
        raise ValueError("description may not be blank")

    vec = embed_texts([description], cfg.embeddings)[0]
    now = _now()
    existing = get_anchor(conn, kind, name)
    created_at = existing.created_at if existing else now
    conn.execute(
        """
        INSERT INTO label_anchors(kind, name, description, embedding,
                                  embed_model, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(kind, name) DO UPDATE SET
            description = excluded.description,
            embedding   = excluded.embedding,
            embed_model = excluded.embed_model,
            updated_at  = excluded.updated_at
        """,
        (kind, name, description, _serialize_vec(vec),
         cfg.embeddings.model, created_at, now),
    )
    result = get_anchor(conn, kind, name)
    assert result is not None
    return result


def delete_anchor(
    conn: sqlite3.Connection, kind: str, name: str
) -> bool:
    cur = conn.execute(
        "DELETE FROM label_anchors WHERE kind = ? AND name = ?", (kind, name)
    )
    return cur.rowcount > 0


def import_anchors(
    conn: sqlite3.Connection,
    cfg: Config,
    anchors: list[dict],
) -> dict:
    """Bulk-upsert. Each item: {kind, name, description}. Returns counts."""
    added = 0
    updated = 0
    errors: list[dict] = []
    for i, item in enumerate(anchors):
        try:
            kind = item.get("kind", "")
            name = item.get("name", "")
            desc = item.get("description", "")
            existed = get_anchor(conn, kind, name) is not None
            set_anchor(conn, cfg, kind, name, desc)
            if existed:
                updated += 1
            else:
                added += 1
        except Exception as e:
            errors.append({"index": i, "item": item, "reason": str(e)})
    return {"added": added, "updated": updated, "errors": errors}


def export_anchors(conn: sqlite3.Connection) -> list[dict]:
    """Portable JSON shape — no embeddings, just source descriptions."""
    return [
        {"kind": a.kind, "name": a.name, "description": a.description}
        for a in list_anchors(conn)
    ]


# ---- scoring --------------------------------------------------------------


def _load_all_embeddings(
    conn: sqlite3.Connection,
) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    """Return (meta, matrix) where matrix is (N, dim) unit-normalized floats.

    Meta is a parallel list of (kind, name, description) so callers can
    map matrix rows back to anchor identity.
    """
    rows = conn.execute(
        "SELECT kind, name, description, embedding FROM label_anchors "
        "ORDER BY kind, name"
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    meta = [(r[0], r[1], r[2]) for r in rows]
    matrix = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    return meta, matrix


def score_anchors(
    conn: sqlite3.Connection,
    doc_vec: np.ndarray,
    min_similarity: float,
) -> list[AnchorMatch]:
    """Cosine-score every anchor against a unit-normalized doc vector.

    Returns only matches at or above ``min_similarity``, sorted descending.
    An empty anchor table returns an empty list — the classifier then
    falls back to neighbor voting.
    """
    meta, matrix = _load_all_embeddings(conn)
    if not meta:
        return []
    # Both sides are unit-normalized (embed_texts sets normalize_embeddings
    # + we re-normalize doc means), so dot product == cosine similarity.
    sims = matrix @ doc_vec
    out: list[AnchorMatch] = []
    for (kind, name, desc), sim in zip(meta, sims, strict=True):
        s = float(sim)
        if s >= min_similarity:
            out.append(AnchorMatch(kind=kind, name=name, similarity=s, description=desc))
    out.sort(key=lambda a: a.similarity, reverse=True)
    return out
