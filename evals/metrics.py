"""Doc-level retrieval metrics and a paired bootstrap.

Conventions follow trec_eval, which is what BEIR numbers are computed with:
linear gain (a grade-2 doc is worth twice a grade-1 doc), a log2(rank + 1)
discount, the ideal DCG cut at k, and "relevant" meaning grade >= 1.

``ranked`` is a list of distinct doc ids, best first; ``qrels`` maps doc id
to grade, with unjudged docs treated as non-relevant. A query with no
relevant docs has no defined score, so every metric returns None for it and
callers leave it out of means (trec_eval skips such queries too). A query
that returns nothing still scores 0.0: an empty result is a miss.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def _relevant(qrels: Mapping[str, int]) -> dict[str, int]:
    return {d: g for d, g in qrels.items() if g >= 1}


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float | None:
    rel = _relevant(qrels)
    if not rel:
        return None
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float | None:
    rel = _relevant(qrels)
    if not rel:
        return None
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def mrr_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float | None:
    rel = _relevant(qrels)
    if not rel:
        return None
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def first_unique(ids: Iterable[str], k: int) -> list[str]:
    """The first ``k`` distinct ids, in order of first appearance."""
    out: list[str] = []
    seen: set[str] = set()
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
            if len(out) == k:
                break
    return out


def paired_bootstrap(
    base: Sequence[float],
    other: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean of ``other - base`` over queries, with a percentile bootstrap CI.

    The two sequences are per-query scores for the same queries in the same
    order; queries are resampled with replacement, keeping each pair
    together. Returns (mean, lo, hi) for a (1 - alpha) interval.
    """
    if len(base) != len(other):
        raise ValueError(f"unpaired scores: {len(base)} vs {len(other)}")
    if not base:
        raise ValueError("no scores to compare")
    diffs = np.asarray(other, dtype=float) - np.asarray(base, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(diffs.mean()), float(lo), float(hi)
