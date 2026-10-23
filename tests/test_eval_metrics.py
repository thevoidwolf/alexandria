"""Retrieval metrics for the eval harness, checked against hand-worked values.

Conventions follow trec_eval (what BEIR reports): linear gain, log2(rank + 1)
discount, ideal DCG cut at k, relevant means grade >= 1. A query with no
relevant documents has no defined score (None), so it can be left out of a
mean instead of dragging it towards zero.
"""
import math
import random

import pytest

from evals.metrics import first_unique, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k

LOG2_3 = math.log2(3)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x"], {"a": 1, "b": 1}) == pytest.approx(1.0)


def test_ndcg_graded_out_of_order():
    # DCG = 1/log2(2) + 2/log2(3); ideal = 2/log2(2) + 1/log2(3).
    got = ndcg_at_k(["b", "a"], {"a": 2, "b": 1})
    assert got == pytest.approx((1 + 2 / LOG2_3) / (2 + 1 / LOG2_3))


def test_ndcg_fewer_results_than_k():
    # Only "a" is found, at rank 2; "b" never comes back.
    got = ndcg_at_k(["x", "a"], {"a": 1, "b": 1}, k=10)
    assert got == pytest.approx((1 / LOG2_3) / (1 + 1 / LOG2_3))


def test_ndcg_ideal_is_cut_at_k():
    # Three relevant docs but k=2: the top two being relevant is already ideal.
    assert ndcg_at_k(["a", "b", "c"], {"a": 1, "b": 1, "c": 1}, k=2) == pytest.approx(1.0)


def test_ndcg_ignores_hits_beyond_k():
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, k=2) == 0.0


def test_recall_counts_graded_docs_and_respects_k():
    ranked = ["x", "a", "y", "z", "w", "b"]
    qrels = {"a": 2, "b": 1, "n": 0}
    assert recall_at_k(ranked, qrels, k=5) == pytest.approx(0.5)
    assert recall_at_k(ranked, qrels, k=10) == pytest.approx(1.0)


def test_recall_grade_zero_is_not_relevant():
    assert recall_at_k(["n"], {"a": 1, "n": 0}, k=5) == 0.0


def test_mrr_first_relevant_rank():
    assert mrr_at_k(["n", "x", "a", "b"], {"a": 1, "b": 2, "n": 0}) == pytest.approx(1 / 3)


def test_mrr_relevant_only_beyond_k():
    assert mrr_at_k(["x", "y", "a"], {"a": 1}, k=2) == 0.0


def test_empty_ranking_scores_zero_when_relevant_docs_exist():
    qrels = {"a": 1}
    assert ndcg_at_k([], qrels) == 0.0
    assert recall_at_k([], qrels, k=10) == 0.0
    assert mrr_at_k([], qrels) == 0.0


@pytest.mark.parametrize("qrels", [{}, {"a": 0, "b": 0}])
def test_no_relevant_docs_is_undefined(qrels):
    assert ndcg_at_k(["a", "b"], qrels) is None
    assert recall_at_k(["a", "b"], qrels, k=10) is None
    assert mrr_at_k(["a", "b"], qrels) is None


def test_first_unique_keeps_first_appearance_order():
    assert first_unique(["a", "b", "a", "c", "b", "d"], 3) == ["a", "b", "c"]
    assert first_unique(["a", "a"], 10) == ["a"]


def test_bootstrap_identical_runs_have_zero_difference():
    xs = [0.1, 0.5, 0.9, 0.0]
    assert paired_bootstrap(xs, list(xs)) == (0.0, 0.0, 0.0)


def test_bootstrap_constant_shift_collapses_the_interval():
    mean, lo, hi = paired_bootstrap([0.2, 0.5, 0.7], [0.3, 0.6, 0.8])
    assert (mean, lo, hi) == pytest.approx((0.1, 0.1, 0.1))


def test_bootstrap_interval_crosses_zero_for_a_wash():
    base = [0.0, 1.0] * 20
    other = [1.0, 0.0] * 20
    mean, lo, hi = paired_bootstrap(base, other)
    assert mean == pytest.approx(0.0)
    assert lo < 0 < hi


def test_bootstrap_consistent_gain_excludes_zero():
    base = [0.0] * 30
    other = [0.1, 0.2, 0.3] * 10
    mean, lo, hi = paired_bootstrap(base, other)
    assert mean == pytest.approx(0.2)
    assert 0 < lo < mean < hi


def test_bootstrap_is_reproducible_with_a_fixed_seed():
    rng = random.Random(1)
    base = [rng.random() for _ in range(50)]
    other = [rng.random() for _ in range(50)]
    assert paired_bootstrap(base, other) == paired_bootstrap(base, other)
    assert paired_bootstrap(base, other, seed=1) != paired_bootstrap(base, other, seed=2)


@pytest.mark.parametrize("base,other", [([0.1, 0.2], [0.1]), ([], [])])
def test_bootstrap_rejects_unpaired_or_empty_input(base, other):
    with pytest.raises(ValueError):
        paired_bootstrap(base, other)


def test_metrics_match_ir_measures():
    ir_measures = pytest.importorskip("ir_measures")
    from ir_measures import RR, R, ScoredDoc, nDCG

    rng = random.Random(7)
    pool = [f"d{i}" for i in range(30)]
    qrels, runs = {}, {}
    for q in range(50):
        qid = f"q{q}"
        qrels[qid] = {d: rng.choice([0, 1, 1, 2]) for d in rng.sample(pool, rng.randint(1, 6))}
        runs[qid] = rng.sample(pool, rng.randint(1, 15))

    ours = {
        (qid, "nDCG@10"): ndcg_at_k(runs[qid], qrels[qid], k=10) for qid in qrels
    } | {
        (qid, "R@5"): recall_at_k(runs[qid], qrels[qid], k=5) for qid in qrels
    } | {
        (qid, "R@10"): recall_at_k(runs[qid], qrels[qid], k=10) for qid in qrels
    } | {
        (qid, "RR@10"): mrr_at_k(runs[qid], qrels[qid], k=10) for qid in qrels
    }
    ir_qrels = [
        ir_measures.Qrel(qid, d, g) for qid, judged in qrels.items() for d, g in judged.items()
    ]
    ir_run = [
        ScoredDoc(qid, d, float(-rank)) for qid, ranked in runs.items()
        for rank, d in enumerate(ranked)
    ]
    theirs = {
        (m.query_id, str(m.measure)): m.value
        for m in ir_measures.iter_calc([nDCG @ 10, R @ 5, R @ 10, RR @ 10], ir_qrels, ir_run)
    }

    compared = 0
    for key, value in ours.items():
        if value is None:
            continue
        assert value == pytest.approx(theirs[key], abs=1e-9), key
        compared += 1
    assert compared > 150
