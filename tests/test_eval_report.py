"""Results -> REPORT.md's generated blocks."""
import pytest

from evals import report as ev_report
from evals.report import (
    Results, by_length, fallback_rows, matched_in_precision, render, top_hit_score,
)


def row(qid, ndcg=0.5, *, n_words=3, n_fts=1, hits=(), latency=10.0):
    return {
        "query_id": qid, "n_words": n_words, "n_fts": n_fts, "latency_ms": latency,
        "docs": [h["doc"] for h in hits], "hits": list(hits),
        "ndcg@10": ndcg, "recall@5": ndcg, "recall@10": ndcg, "mrr@10": ndcg,
    }


def hit(doc, fts=None, vec=None):
    return {"doc": doc, "chunk": 0, "fts_rank": fts, "vec_rank": vec,
            "matched_in": [m for m, r in (("fts", fts), ("vec", vec)) if r is not None]}


def results(rows, *, pooled=False, judged=1.0, qrels=None, meta=None, name="tiny"):
    variants = {v: {"judged@10": judged} if pooled else {} for v in rows}
    for v in variants.values():
        v["search"] = {"rrf_k": 60}
    queries = {q: q for v in rows.values() for q in v}
    return Results(
        name=name, summary={"baseline": "hybrid", "variants": variants}, rows=rows,
        queries=queries, query_meta=meta or {}, qrels=qrels or {}, pooled=pooled, tokens=None,
    )


REPORT = """# Report

Hand-written text.

<!-- report:demo tiny -->
old
<!-- /report -->

More text.
"""


def test_render_replaces_only_what_is_between_the_markers(monkeypatch):
    monkeypatch.setitem(ev_report.BLOCKS, "demo", (lambda r: f"new for {r.name}", False))
    out, skipped = render(REPORT, {"tiny": results({"hybrid": {}})})
    assert out == REPORT.replace("old", "new for tiny")
    assert skipped == []


def test_render_refuses_unknown_blocks_datasets_and_unclosed_markers():
    with pytest.raises(ValueError, match="nonesuch"):
        render("<!-- report:nonesuch tiny -->\n<!-- /report -->\n", {})
    with pytest.raises(ValueError, match="other"):
        render("<!-- report:main other -->\n<!-- /report -->\n", {"tiny": results({})})
    with pytest.raises(ValueError, match="unclosed"):
        render("<!-- report:main tiny -->\nno end\n", {"tiny": results({})})


def test_text_that_mentions_the_marker_syntax_is_not_a_block(monkeypatch):
    monkeypatch.setitem(ev_report.BLOCKS, "demo", (lambda r: "x", False))
    text = "Tables between `<!-- report:… -->` markers are generated.\n\n" + REPORT
    out, _ = render(text, {"tiny": results({"hybrid": {}})})
    assert out.startswith("Tables between `<!-- report:… -->` markers")


def test_blocks_built_on_grades_wait_until_every_variant_is_judged(monkeypatch):
    monkeypatch.setitem(ev_report.BLOCKS, "demo", (lambda r: "graded", True))
    half = {"tiny": results({"hybrid": {}}, pooled=True, judged=0.7)}
    out, skipped = render(REPORT, half)
    assert out == REPORT
    assert skipped == ["demo tiny"]

    out, _ = render(REPORT, half, provisional=True)
    assert "graded" in out and "Provisional" in out

    out, skipped = render(REPORT, {"tiny": results({"hybrid": {}}, pooled=True, judged=1.0)})
    assert "graded" in out and "Provisional" not in out and skipped == []


def test_blocks_that_use_no_grades_render_before_judging_is_done(monkeypatch):
    monkeypatch.setitem(ev_report.BLOCKS, "demo", (lambda r: "counts", False))
    out, skipped = render(REPORT, {"tiny": results({"hybrid": {}}, pooled=True, judged=0.5)})
    assert "counts" in out and skipped == []


def test_queries_are_bucketed_by_length():
    rows = {q: row(q, n_words=n) for q, n in [("a", 1), ("b", 4), ("c", 5), ("d", 12), ("e", 13)]}
    assert {b: sorted(r["query_id"] for r in rs) for b, rs in by_length(rows).items()} == {
        "1–4": ["a", "b"], "5–8": ["c"], "9–12": ["d"], "13+": ["e"],
    }


def test_fallback_takes_or_rows_only_where_and_found_nothing():
    hybrid = {"q1": row("q1", 0.4, n_fts=3), "q2": row("q2", 0.2, n_fts=0)}
    hybrid_or = {"q1": row("q1", 0.9, n_fts=50), "q2": row("q2", 0.6, n_fts=40)}
    out = fallback_rows(hybrid, hybrid_or)
    assert out["q1"] is hybrid["q1"]
    assert out["q2"] is hybrid_or["q2"]


def test_precision_by_retriever_counts_each_returned_paper_by_its_best_chunk():
    rows = {
        "q1": row("q1", hits=[hit("a", fts=1, vec=1), hit("b", vec=2), hit("c", fts=2)]),
        "q2": row("q2", hits=[hit("a", fts=1, vec=3), hit("x", vec=1)]),
    }
    qrels = {"q1": {"a": 2, "b": 0}, "q2": {"x": 1}}     # c is unjudged: not relevant
    got = matched_in_precision(rows, qrels)
    assert got["both"]["all"] == (2, 1)                   # (hits, relevant)
    assert got["vec"]["all"] == (2, 1)
    assert got["fts"]["all"] == (1, 0)
    assert got["both"]["1–3"] == (2, 1) and got["both"]["4–10"] == (0, 0)


def test_precision_gap_compares_both_with_single_retriever_hits_by_query_bootstrap():
    rows = {
        f"q{i}": row(f"q{i}", hits=[hit("a", fts=1, vec=1), hit("b", vec=2), hit("c", vec=3)])
        for i in range(20)
    }
    qrels = {q: {"a": 1, "b": 1 if i % 2 else 0} for i, q in enumerate(rows)}
    diff, lo, hi = ev_report.precision_gap(rows, qrels, 0, 3)
    assert diff == pytest.approx(1 - 10 / 40)            # 20/20 both, 10/40 single
    assert lo <= diff <= hi and lo > 0
    assert ev_report.precision_gap(rows, qrels, 3, 10) is None   # no hits at ranks 4-10
    few = {q: rows[q] for q in list(rows)[:4]}                   # 4 both, 8 single
    assert ev_report.precision_gap(few, qrels, 0, 3) is None


def test_main_table_keeps_the_variant_order_and_bolds_only_real_differences():
    def entry(ndcg, diff=None):
        e = {"metrics": {m: {"mean": ndcg, "n": 2} for m in
                         ("ndcg@10", "recall@5", "recall@10", "mrr@10")},
             "latency_ms": {"p50": 12.34, "p95": 11940.0}}
        if diff:
            e["vs_baseline"] = {"ndcg@10": dict(zip(("diff", "lo", "hi"), diff))}
        return e

    r = results({"vec": {}, "hybrid": {}, "fts": {}})
    r.summary["variants"] = {"vec": entry(0.7, (-0.002, -0.006, 0.001)), "hybrid": entry(0.72),
                             "fts": entry(0.05, (-0.675, -0.72, -0.63))}
    r.queries = {"q1": "", "q2": "", "q3": ""}
    lines = ev_report.main_table(r).splitlines()
    assert [line.split(" | ")[0] for line in lines[2:5]] == ["| `fts`", "| `vec`", "| `hybrid`"]
    assert "**−0.675 [−0.720, −0.630]**" in lines[2]
    assert "−0.002 [−0.006, +0.001]" in lines[3] and "**" not in lines[3]
    assert "12.3 / 11,940" in lines[4]
    assert "Means over the 2 of 3 queries" in lines[-1]


def test_top_hit_score_is_the_fused_score_of_the_first_chunk():
    assert top_hit_score(row("q", hits=[hit("a", fts=1, vec=3), hit("b", vec=1)]), 60) == \
        pytest.approx(1 / 61 + 1 / 63)
    assert top_hit_score(row("q", hits=[hit("a", vec=2)]), 60) == pytest.approx(1 / 62)
    assert top_hit_score(row("q", hits=[]), 60) is None
