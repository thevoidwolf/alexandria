from __future__ import annotations

from functools import lru_cache

import numpy as np

from alexandria.config import EmbeddingsConfig


@lru_cache(maxsize=4)
def _get_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def embed_texts(texts: list[str], cfg: EmbeddingsConfig) -> np.ndarray:
    """Return an (N, dim) float32 array. Normalized for cosine similarity."""
    if not texts:
        return np.zeros((0, cfg.dim), dtype=np.float32)
    model = _get_model(cfg.model, cfg.device)
    vecs = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32)
