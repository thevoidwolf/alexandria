"""Cross-encoder reranking for search (opt-in; see ``SearchConfig``)."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _get_cross_encoder(model_name: str, device: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


def rerank_scores(query: str, texts: list[str], model: str, device: str) -> list[float]:
    """Score each text's relevance to the query; higher is more relevant.

    The model reads query and text together, truncated to its input window
    (512 tokens for bge-reranker-base).
    """
    if not texts:
        return []
    scores = _get_cross_encoder(model, device).predict(
        [(query, t) for t in texts], batch_size=32, show_progress_bar=False,
    )
    return [float(s) for s in scores]
