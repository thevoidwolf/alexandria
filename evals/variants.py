"""Named retrieval variants, declared as data.

A variant is what changes relative to the default install: the search mode,
the index settings (which need their own index) or query-side settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from alexandria.search import SearchMode
from evals.index import IndexSettings


@dataclass(frozen=True)
class Variant:
    name: str
    mode: SearchMode = "hybrid"
    index: IndexSettings = field(default_factory=IndexSettings)
    query_instruction: str = ""


_ALL = [
    Variant("fts", mode="fts"),
    Variant("vec", mode="vec"),
    Variant("hybrid"),           # the default install
]
VARIANTS: dict[str, Variant] = {v.name: v for v in _ALL}
BASELINE = "hybrid"


def resolve(spec: str) -> list[Variant]:
    """``"all"`` or a comma-separated list of variant names."""
    if spec.strip() == "all":
        return list(_ALL)
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variant(s): {', '.join(unknown)}; known: {', '.join(VARIANTS)}")
    return [VARIANTS[n] for n in names]
