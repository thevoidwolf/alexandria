"""Blind pooled judging: pool every variant's top 10, hide who found what."""
import pytest

from evals.datasets import Dataset
from evals.index import IndexSettings, build_index
from evals.judge import Pair, best_window, check_fresh, judge, pool, session
from evals.run import run_variant
from evals.variants import Variant

SMALL = IndexSettings(chunk_tokens=50, chunk_overlap=10)


@pytest.fixture(autouse=True)
def _no_hub_lookup(monkeypatch):
    from evals import index as ev_index

    monkeypatch.setattr(ev_index, "model_revision", lambda name: "test-rev")


def _row(qid, *hits):
    return {"query_id": qid, "docs": [d for d, _ in hits],
            "hits": [{"doc": d, "chunk": c} for d, c in hits]}


ROWS = {
    "variant-alpha": [_row("q1", ("a", 0), ("b", 2)), _row("q2", ("c", 0))],
    "variant-beta": [_row("q1", ("b", 1), ("d", 0)), _row("q2", ("c", 0))],
}


def test_pool_unions_variants_and_skips_judged_pairs():
    pairs = pool(ROWS, {"q1": {"d": 0}}, ["q1", "q2"])
    assert sorted((p.query_id, p.doc) for p in pairs) == [("q1", "a"), ("q1", "b"), ("q2", "c")]
    assert [p.query_id for p in pairs] == ["q1", "q1", "q2"]          # query order kept


def test_pool_keeps_every_chunk_that_surfaced_a_doc_best_rank_first():
    b = next(p for p in pool(ROWS, {}, ["q1"]) if p.doc == "b")
    assert b.chunks == (("variant-beta", 1), ("variant-alpha", 2))


def test_pool_order_is_shuffled_reproducibly():
    rows = {"v": [_row("q", *((f"d{i}", 0) for i in range(10)))]}
    first = [p.doc for p in pool(rows, {}, ["q"], seed=3)]
    assert first == [p.doc for p in pool(rows, {}, ["q"], seed=3)]
    assert sorted(first) == sorted(f"d{i}" for i in range(10))
    assert first != [f"d{i}" for i in range(10)]                       # not rank order


def test_pool_ignores_rows_for_queries_no_longer_in_the_set():
    assert [p.query_id for p in pool(ROWS, {}, ["q2"])] == ["q2"]


def test_best_window_finds_the_query_words():
    words = [f"filler{i}" for i in range(200)]
    words[150:153] = ["refusal", "rate", "benchmark"]
    window = best_window(" ".join(words), "over-refusal benchmark rate", width=20)
    assert "refusal rate benchmark" in window
    assert window.startswith("…") and window.endswith("…")


def test_best_window_returns_short_text_whole():
    assert best_window("a short chunk", "chunk", width=20) == "a short chunk"


CHUNKS = {("variant-alpha", 0): "alpha text " * 5 + "TAIL-ONLY-IN-FULL",
          ("variant-beta", 1): "beta text"}


def _judge(keys, pairs=None):
    shown, recorded = [], []
    answers = iter(keys)

    def ask(prompt):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    n = judge(
        pairs or [Pair("q1", "a", (("variant-alpha", 0),)), Pair("q1", "b", (("variant-beta", 1),))],
        query_text={"q1": "what is alpha"},
        title={"a": "Paper A", "b": "Paper B"},
        chunk_text=lambda variant, doc, ord_: CHUNKS[(variant, ord_)],
        ask=ask, show=shown.append, record=lambda p, g: recorded.append((p.doc, g)),
        window=3,
    )
    return n, recorded, "\n".join(shown)


def test_judge_records_grades_and_reasks_on_bad_input():
    n, recorded, out = _judge(["7", "2", "0"])
    assert n == 2 and recorded == [("a", 2), ("b", 0)]
    assert "Paper A" in out and "what is alpha" in out


def test_judge_never_shows_which_variant_found_a_doc():
    _, _, out = _judge(["m", "1", "1"])
    assert "variant-alpha" not in out and "variant-beta" not in out


def test_judge_more_shows_the_whole_chunk():
    _, _, short = _judge(["1", "1"])
    _, _, full = _judge(["m", "1", "1"])
    assert "TAIL-ONLY-IN-FULL" not in short and "TAIL-ONLY-IN-FULL" in full


def test_judge_skip_leaves_a_pair_unjudged_and_quit_stops():
    n, recorded, _ = _judge(["s", "q"])
    assert n == 0 and recorded == []
    n, recorded, _ = _judge(["1", "q"])
    assert n == 1 and recorded == [("a", 1)]


def test_judge_end_of_input_stops_cleanly():
    n, recorded, _ = _judge(["2"])
    assert n == 1 and recorded == [("a", 2)]


@pytest.fixture
def tiny():
    return Dataset(
        name="tiny",
        docs={"a": "Reciprocal rank fusion merges keyword and vector rankings.",
              "b": "Over-refusal benchmarks measure how often models decline safe requests.",
              "c": "The water bill for March was unusually high."},
        queries={"q1": "rank fusion", "q2": "over-refusal benchmark"},
        qrels={},
    )


def _read_qrels(path):
    out = {}
    for line in path.read_text().splitlines():
        q, d, g = line.split("\t")
        out.setdefault(q, {})[d] = int(g)
    return out


def test_an_interrupted_session_resumes_where_it_stopped(tiny, tmp_path, fake_embed):
    built = build_index(tiny, SMALL, tmp_path / "cache")
    rows = {"hybrid": run_variant(tiny, Variant("hybrid", index=SMALL), built)}
    qrels_path = tmp_path / "qrels.tsv"
    total = len(pool(rows, {}, list(tiny.queries)))
    seen = []

    def ask_then_quit(keys):
        answers = iter(keys)
        return lambda prompt: next(answers)

    def show(text):
        seen.append(text)

    first = session(tiny, rows, {"hybrid": built}, qrels_path, ask=ask_then_quit(["2", "q"]), show=show)
    assert first == 1
    judged = _read_qrels(qrels_path)
    assert sum(len(v) for v in judged.values()) == 1

    resumed = Dataset(**{**tiny.__dict__, "qrels": judged})
    left = pool(rows, judged, list(tiny.queries))
    assert len(left) == total - 1
    second = session(resumed, rows, {"hybrid": built}, qrels_path,
                     ask=ask_then_quit(["1"] * len(left)), show=show)
    assert second == total - 1
    assert sum(len(v) for v in _read_qrels(qrels_path).values()) == total


def test_results_from_a_different_index_are_refused():
    check_fresh({"hybrid": {"index": {"key": "tiny-1111"}}}, {"hybrid": "tiny-1111"})
    with pytest.raises(ValueError, match="hybrid.*evals.run"):
        check_fresh({"hybrid": {"index": {"key": "tiny-0000"}}}, {"hybrid": "tiny-1111"})
    with pytest.raises(ValueError, match="hybrid"):
        check_fresh({}, {"hybrid": "tiny-1111"})          # rows on disk but no summary entry
