"""Full-run cross-check: recompute a run's per-query metrics with ir-measures."""
import pytest

pytest.importorskip("ir_measures")

from evals.crosscheck import compare  # noqa: E402

QRELS = {"q1": {"a": 2, "b": 1}, "q2": {"c": 1}, "q3": {"a": 0}}


def _row(qid, docs, **metrics):
    return {"query_id": qid, "docs": docs, **metrics}


def _good_rows():
    return [
        # DCG = 1 + 2/log2(3); ideal = 2 + 1/log2(3)
        _row("q1", ["b", "a"], **{"ndcg@10": 0.8597186998521972, "recall@5": 1.0,
                                  "recall@10": 1.0, "mrr@10": 1.0}),
        # Empty result: ir-measures never sees q2, which must still be a 0.
        _row("q2", [], **{"ndcg@10": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0}),
        _row("q3", ["a"], **{"ndcg@10": None, "recall@5": None, "recall@10": None, "mrr@10": None}),
    ]


def test_matching_run_has_no_differences():
    result = compare(_good_rows(), QRELS)
    assert result["max_abs_diff"] == pytest.approx(0.0, abs=1e-12)
    assert result["n_compared"] == 8          # q1 and q2 x 4 metrics; q3 undefined


def test_a_wrong_score_is_reported():
    rows = _good_rows()
    rows[0]["mrr@10"] = 0.5
    result = compare(rows, QRELS)
    assert result["max_abs_diff"] == pytest.approx(0.5)
    assert result["worst"] == {"query_id": "q1", "metric": "mrr@10"}


def test_a_nonzero_score_for_an_empty_result_is_reported():
    rows = _good_rows()
    rows[1]["recall@10"] = 1.0
    assert compare(rows, QRELS)["max_abs_diff"] == pytest.approx(1.0)
