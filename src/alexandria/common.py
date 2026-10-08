"""Helpers shared by the MCP tools, the web API/pages and the CLI.

Serializers live here so the MCP and HTTP responses can't drift apart.
Pure Python (no Starlette), so the stdio MCP server can import it.
"""
from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any

from alexandria.anchors import Anchor
from alexandria.classify import Suggestion
from alexandria.search import SearchHit, format_snippet_markdown


def package_version() -> str:
    try:
        return importlib_metadata.version("alexandria")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0+dev"


def split_tags(raw: str | None) -> list[str]:
    """Parse a comma-separated tag string, dropping blanks."""
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def search_hit_payload(h: SearchHit) -> dict[str, Any]:
    return {
        "chunk_id": h.chunk_id,
        "doc_id": h.doc_id,
        "score": h.score,
        "fts_rank": h.fts_rank,
        "vec_rank": h.vec_rank,
        "matched_in": list(h.matched_in),
        "snippet": format_snippet_markdown(h.snippet),
        "title": h.title,
        "display_title": h.display_title,
        "category": h.category,
        "tags": h.tags,
        "source_uri": h.source_uri,
        "content_type": h.content_type,
    }


def suggestion_payload(s: Suggestion, applied: bool | None = None) -> dict[str, Any]:
    """``applied`` overrides ``s.applied`` when the caller applied it itself."""
    return {
        "doc_id": s.doc_id,
        "current": s.current,
        "suggested_category": s.suggested_category,
        "suggested_tags": list(s.suggested_tags),
        "category_confidence": s.category_confidence,
        "tag_confidences": dict(s.tag_confidences),
        "category_source": s.category_source,
        "tag_source": s.tag_source,
        "anchor_matches": [
            {
                "kind": a.kind,
                "name": a.name,
                "similarity": a.similarity,
                "description": a.description,
            }
            for a in s.anchor_matches
        ],
        "neighbors": [
            {
                "doc_id": n.doc_id,
                "display_title": n.display_title,
                "similarity": n.similarity,
                "category": n.category,
                "tags": list(n.tags),
            }
            for n in s.neighbors
        ],
        "applied": s.applied if applied is None else applied,
    }


def anchor_payload(a: Anchor) -> dict[str, Any]:
    return {
        "kind": a.kind,
        "name": a.name,
        "description": a.description,
        "embed_model": a.embed_model,
        "created_at": a.created_at,
        "updated_at": a.updated_at,
    }
