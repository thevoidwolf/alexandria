"""Named retrieval variants, declared as data.

A variant is what changes relative to the default install: the search mode,
the index settings (which need their own index) or query-side settings.
The table is the brief's (§5), with the changes recorded in REPORT.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from alexandria.search import RRF_K, FtsOperator, SearchMode
from evals.index import IndexSettings

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_BASE = IndexSettings(embed_model="BAAI/bge-base-en-v1.5", embed_dim=768)


@dataclass(frozen=True)
class Variant:
    name: str
    mode: SearchMode = "hybrid"
    index: IndexSettings = field(default_factory=IndexSettings)
    query_instruction: str = ""
    rrf_k: int = RRF_K
    fts_operator: FtsOperator = "and"
    rerank: bool = False
    titles: bool = False
    datasets: tuple[str, ...] | None = None    # None: every dataset


_ALL = [
    Variant("fts", mode="fts"),
    Variant("vec", mode="vec"),
    Variant("hybrid"),           # the default install
    Variant("hybrid+qi", query_instruction=QUERY_INSTRUCTION),
    Variant("hybrid+rerank", rerank=True),
    Variant("vec-base", mode="vec", index=BGE_BASE),
    Variant("hybrid-base", index=BGE_BASE),
    # SciFact abstracts are short, so chunk size only matters for papers.
    Variant("hybrid-c300", index=IndexSettings(chunk_tokens=300, chunk_overlap=50),
            datasets=("papers",)),
    Variant("hybrid-c200", index=IndexSettings(chunk_tokens=200, chunk_overlap=40),
            datasets=("papers",)),
    Variant("hybrid-k20", rrf_k=20),          # a sensitivity check, not a tuning target
    Variant("fts-or", mode="fts", fts_operator="or"),
    Variant("hybrid-or", fts_operator="or"),
    # H7: SciFact is ingested without titles, so there it is exactly hybrid.
    Variant("hybrid+title", titles=True, datasets=("papers", "papers-titles")),
]
VARIANTS: dict[str, Variant] = {v.name: v for v in _ALL}
BASELINE = "hybrid"


def resolve(spec: str, dataset: str) -> list[Variant]:
    """``"all"`` (every variant that applies to ``dataset``) or a
    comma-separated list of variant names."""
    def applies(v: Variant) -> bool:
        return v.datasets is None or dataset in v.datasets

    if spec.strip() == "all":
        return [v for v in _ALL if applies(v)]
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variant(s): {', '.join(unknown)}; known: {', '.join(VARIANTS)}")
    elsewhere = [n for n in names if not applies(VARIANTS[n])]
    if elsewhere:
        raise ValueError(f"{', '.join(elsewhere)} doesn't apply to {dataset}")
    return [VARIANTS[n] for n in names]
