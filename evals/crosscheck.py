"""Recompute a run's per-query metrics with ir-measures and compare.

    uv run python -m evals.crosscheck --dataset scifact

Our metrics are implemented in evals/metrics.py; this confirms a full run
agrees with an independent implementation (pytrec_eval, via ir-measures).
A query with no results never reaches ir-measures, so it counts as 0 there,
as trec_eval -c does. Exits non-zero on any difference above 1e-9.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ir_measures
from ir_measures import RR, R, ScoredDoc, nDCG

from evals.index import build_index, translate_qrels
from evals.run import DATASETS, METRICS, REPO
from evals.variants import VARIANTS

TOLERANCE = 1e-9
_MEASURES = {"ndcg@10": nDCG @ 10, "recall@5": R @ 5, "recall@10": R @ 10, "mrr@10": RR @ 10}


def compare(rows: list[dict], qrels: dict[str, dict[str, int]]) -> dict:
    """Largest |ours - ir_measures| over every defined per-query metric."""
    ir_qrels = [
        ir_measures.Qrel(qid, d, g) for qid, judged in qrels.items() for d, g in judged.items()
    ]
    ir_run = [
        ScoredDoc(r["query_id"], d, float(-rank))
        for r in rows for rank, d in enumerate(r["docs"])
    ]
    by_name = {str(m): name for name, m in _MEASURES.items()}
    theirs = {
        (m.query_id, by_name[str(m.measure)]): m.value
        for m in ir_measures.iter_calc(list(_MEASURES.values()), ir_qrels, ir_run)
    }
    worst, max_diff, n = None, 0.0, 0
    for r in rows:
        for metric in METRICS:
            if r[metric] is None:
                continue
            diff = abs(r[metric] - theirs.get((r["query_id"], metric), 0.0))
            n += 1
            if diff > max_diff:
                max_diff, worst = diff, {"query_id": r["query_id"], "metric": metric}
    return {"max_abs_diff": max_diff, "n_compared": n, "worst": worst}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m evals.crosscheck")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--results", type=Path, default=REPO / "evals" / "results")
    p.add_argument("--cache", type=Path, default=REPO / "evals" / ".cache")
    args = p.parse_args(argv)

    ds = DATASETS[args.dataset](args.cache)
    failed = False
    for path in sorted((args.results / ds.name).glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        built = build_index(ds, VARIANTS[path.stem].index, args.cache)
        result = compare(rows, translate_qrels(ds.qrels, built.doc_map))
        ok = result["max_abs_diff"] <= TOLERANCE
        failed |= not ok
        print(f"{path.stem:<14} {'OK ' if ok else 'MISMATCH'} "
              f"max |diff| = {result['max_abs_diff']:.2e} over {result['n_compared']} values"
              + ("" if ok else f" (worst: {result['worst']})"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
