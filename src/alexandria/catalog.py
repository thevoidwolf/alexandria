from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class DocumentRow:
    id: str
    content_type: str
    title: str | None
    author: str | None
    published_at: str | None
    category: str | None
    bytes: int
    ingested_at: str
    last_seen_at: str
    tags: list[str]
    sources: list[dict]                 # [{source_kind, source_uri, seen_at}, ...]
    extracted_text: str | None = None


@dataclass
class CatalogSummary:
    total_docs: int
    total_bytes: int
    categories: list[dict]              # [{name, count}, ...]
    tags: list[dict]                    # [{name, count}, ...]


def _load_tags(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT tag FROM tags WHERE doc_id = ? ORDER BY tag", (doc_id,)
    )]


def _load_sources(conn: sqlite3.Connection, doc_id: str) -> list[dict]:
    return [
        {"source_kind": r[0], "source_uri": r[1], "seen_at": r[2]}
        for r in conn.execute(
            "SELECT source_kind, source_uri, seen_at FROM document_sources "
            "WHERE doc_id = ? ORDER BY seen_at",
            (doc_id,),
        )
    ]


def get_document(
    doc_id: str, conn: sqlite3.Connection, include_text: bool = False
) -> DocumentRow | None:
    cols = (
        "id, content_type, title, author, published_at, category, bytes, "
        "ingested_at, last_seen_at"
    )
    if include_text:
        cols += ", extracted_text"
    row = conn.execute(
        f"SELECT {cols} FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        return None

    return DocumentRow(
        id=row[0],
        content_type=row[1],
        title=row[2],
        author=row[3],
        published_at=row[4],
        category=row[5],
        bytes=row[6],
        ingested_at=row[7],
        last_seen_at=row[8],
        extracted_text=row[9] if include_text else None,
        tags=_load_tags(conn, doc_id),
        sources=_load_sources(conn, doc_id),
    )


def list_documents(
    conn: sqlite3.Connection,
    category: str | None = None,
    tags: list[str] | None = None,
    since: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[DocumentRow]:
    """List document metadata (no text body), filtered and paged."""
    params: list = []
    sql = (
        "SELECT d.id, d.content_type, d.title, d.author, d.published_at, "
        "d.category, d.bytes, d.ingested_at, d.last_seen_at FROM documents d"
    )

    if tags:
        placeholders = ",".join("?" * len(tags))
        sql += (
            f" JOIN (SELECT doc_id FROM tags WHERE tag IN ({placeholders}) "
            f"GROUP BY doc_id HAVING COUNT(DISTINCT tag) = ?) t ON t.doc_id = d.id"
        )
        params.extend(tags)
        params.append(len(tags))

    where: list[str] = []
    if category:
        where.append("d.category = ?")
        params.append(category)
    if since:
        where.append("d.ingested_at >= ?")
        params.append(since)
    if where:
        sql += " WHERE " + " AND ".join(where)

    sql += " ORDER BY d.ingested_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()
    result: list[DocumentRow] = []
    for r in rows:
        doc_id = r[0]
        result.append(
            DocumentRow(
                id=doc_id,
                content_type=r[1],
                title=r[2],
                author=r[3],
                published_at=r[4],
                category=r[5],
                bytes=r[6],
                ingested_at=r[7],
                last_seen_at=r[8],
                tags=_load_tags(conn, doc_id),
                sources=_load_sources(conn, doc_id),
            )
        )
    return result


def get_catalog(conn: sqlite3.Connection) -> CatalogSummary:
    total_docs, total_bytes = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM documents"
    ).fetchone()
    categories = [
        {"name": row[0], "count": row[1]}
        for row in conn.execute(
            "SELECT category, COUNT(*) FROM documents "
            "WHERE category IS NOT NULL GROUP BY category ORDER BY COUNT(*) DESC"
        )
    ]
    tag_rows = [
        {"name": row[0], "count": row[1]}
        for row in conn.execute(
            "SELECT tag, COUNT(*) FROM tags GROUP BY tag ORDER BY COUNT(*) DESC"
        )
    ]
    return CatalogSummary(
        total_docs=total_docs,
        total_bytes=total_bytes,
        categories=categories,
        tags=tag_rows,
    )
