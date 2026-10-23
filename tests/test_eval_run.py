"""Eval runner: variants x dataset -> per-query rows and a summary."""
import json
import subprocess

import pytest

from evals import run as ev_run
from evals.datasets import Dataset
from evals.index import IndexSettings, build_index
from evals.run import run_variant, summarize
from evals.variants import Variant, resolve

SMALL = IndexSettings(chunk_tokens=50, chunk_overlap=10)


@pytest.fixture(autouse=True)
def _no_hub_lookup(monkeypatch):
    from evals import index as ev_index

    monkeypatch.setattr(ev_index, "model_revision", lambda name: "test-rev")
    monkeypatch.setattr(ev_run, "rerank_revision", lambda name: "test-rerank-rev")


@pytest.fixture
def tiny() -> Dataset:
    return Dataset(
        name="tiny",
        docs={
            "a": "Reciprocal rank fusion merges keyword and vector rankings.",
            "c": "The water bill for March was unusually high.",
        },
        queries={
            "q1": "water bill",
            "q2": "fusion zebra",      # "zebra" is in no doc: FTS (implicit AND) finds nothing
            "q3": "rank fusion",       # judged, but nothing relevant: undefined
        },
        qrels={"q1": {"c": 1}, "q2": {"a": 1}, "q3": {"a": 0}},
    )


def _run(ds, tmp_path, **kw):
    v = Variant(kw.pop("name", "x"), index=SMALL, **kw)
    return run_variant(ds, v, build_index(ds, SMALL, tmp_path / "cache"))


def test_rows_are_keyed_by_dataset_ids(tiny, tmp_path, fake_embed):
    rows = {r["query_id"]: r for r in _run(tiny, tmp_path, mode="fts")}
    assert rows["q1"]["docs"] == ["c"]
    assert rows["q1"]["hits"] == [
        {"doc": "c", "chunk": 0, "fts_rank": 1, "vec_rank": None, "matched_in": ["fts"]}
    ]
    assert rows["q1"]["ndcg@10"] == pytest.approx(1.0)
    assert rows["q1"]["n_words"] == 2


def test_zero_hit_query_is_a_miss_not_a_gap(tiny, tmp_path, fake_embed):
    rows = _run(tiny, tmp_path, name="fts", mode="fts")
    by_id = {r["query_id"]: r for r in rows}
    assert by_id["q2"]["docs"] == []
    assert by_id["q2"]["n_fts"] == 0
    assert by_id["q2"]["ndcg@10"] == 0.0

    summary = summarize(tiny, {"fts": rows}, {"fts": {}})
    ndcg = summary["variants"]["fts"]["metrics"]["ndcg@10"]
    assert ndcg["n"] == 2                       # q1 and q2; q3 is undefined
    assert ndcg["mean"] == pytest.approx(0.5)
    assert summary["variants"]["fts"]["fts_zero_hit_rate"] == pytest.approx(1 / 3)


def test_query_with_nothing_relevant_is_undefined(tiny, tmp_path, fake_embed):
    rows = {r["query_id"]: r for r in _run(tiny, tmp_path, mode="hybrid")}
    assert rows["q3"]["ndcg@10"] is None
    assert rows["q3"]["mrr@10"] is None


def test_only_non_baseline_variants_get_a_comparison(tiny, tmp_path, fake_embed):
    built = build_index(tiny, SMALL, tmp_path / "cache")
    rows = {
        "fts": run_variant(tiny, Variant("fts", mode="fts", index=SMALL), built),
        "hybrid": run_variant(tiny, Variant("hybrid", index=SMALL), built),
    }
    summary = summarize(tiny, rows, {"fts": built.stats, "hybrid": built.stats})
    diff = summary["variants"]["fts"]["vs_baseline"]["ndcg@10"]
    assert set(diff) == {"diff", "lo", "hi", "n"}
    assert "vs_baseline" not in summary["variants"]["hybrid"]


def test_resolve_variant_lists():
    assert [v.name for v in resolve("all", "scifact")][:3] == ["fts", "vec", "hybrid"]
    assert [v.name for v in resolve("vec, fts", "scifact")] == ["vec", "fts"]
    with pytest.raises(ValueError, match="nope"):
        resolve("fts,nope", "scifact")


def test_main_writes_rows_and_summary(tiny, tmp_path, fake_embed, monkeypatch):
    monkeypatch.setitem(ev_run.DATASETS, "tiny", lambda cache_dir: tiny)
    monkeypatch.setattr(ev_run, "resolve", lambda spec, dataset: [
        Variant("fts", mode="fts", index=SMALL), Variant("hybrid", index=SMALL),
    ])
    out = tmp_path / "results"
    ev_run.main(["--dataset", "tiny", "--out", str(out), "--cache", str(tmp_path / "cache")])

    lines = (out / "tiny" / "fts.jsonl").read_text().splitlines()
    assert [json.loads(line)["query_id"] for line in lines] == ["q1", "q2", "q3"]
    summary = json.loads((out / "tiny" / "summary.json").read_text())
    assert set(summary["variants"]) == {"fts", "hybrid"}
    assert summary["variants"]["hybrid"]["index"]["n_docs"] == 2


def test_main_records_the_environment_from_before_the_run(tiny, tmp_path, fake_embed, monkeypatch):
    # A commit made while a long run is going must not be credited with its results.
    head = {"alexandria_commit": "start"}
    real_run_variant = ev_run.run_variant

    def run_then_commit(*a, **kw):
        rows = real_run_variant(*a, **kw)
        head["alexandria_commit"] = "committed mid-run"
        return rows

    monkeypatch.setattr(ev_run, "environment", lambda: dict(head))
    monkeypatch.setattr(ev_run, "run_variant", run_then_commit)
    monkeypatch.setitem(ev_run.DATASETS, "tiny", lambda cache_dir: tiny)
    monkeypatch.setattr(ev_run, "resolve", lambda spec, dataset: [Variant("fts", mode="fts", index=SMALL)])
    out = tmp_path / "results"
    ev_run.main(["--dataset", "tiny", "--out", str(out), "--cache", str(tmp_path / "cache")])
    summary = json.loads((out / "tiny" / "summary.json").read_text())
    assert summary["variants"]["fts"]["environment"] == {"alexandria_commit": "start"}


def _main(tiny, tmp_path, monkeypatch, variants):
    monkeypatch.setitem(ev_run.DATASETS, "tiny", lambda cache_dir: tiny)
    monkeypatch.setattr(ev_run, "resolve", lambda spec, dataset: variants)
    out = tmp_path / "results"
    ev_run.main(["--dataset", "tiny", "--out", str(out), "--cache", str(tmp_path / "cache")])
    return json.loads((out / "tiny" / "summary.json").read_text())


def test_a_subset_run_keeps_the_other_variants_in_the_summary(tiny, tmp_path, fake_embed, monkeypatch):
    fts, hybrid = Variant("fts", mode="fts", index=SMALL), Variant("hybrid", index=SMALL)
    first = _main(tiny, tmp_path, monkeypatch, [fts, hybrid])
    again = _main(tiny, tmp_path, monkeypatch, [fts])
    assert set(again["variants"]) == {"fts", "hybrid"}
    assert again["variants"]["hybrid"] == first["variants"]["hybrid"]
    assert "ndcg@10" in again["variants"]["fts"]["vs_baseline"]


def test_table_does_not_call_an_uncompared_variant_the_baseline(tiny, tmp_path, fake_embed):
    rows = _run(tiny, tmp_path, name="fts", mode="fts")
    table = ev_run.format_table(summarize(tiny, {"fts": rows}, {"fts": {}}))
    assert "baseline" not in table.splitlines()[-1]


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True,
    )


def test_commit_is_dirty_only_for_changes_outside_the_results(tmp_path):
    repo = tmp_path / "repo"
    (repo / "evals" / "results").mkdir(parents=True)
    (repo / "code.py").write_text("x = 1\n")
    (repo / "evals" / "results" / "fts.jsonl").write_text("{}\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")

    (repo / "evals" / "results" / "fts.jsonl").write_text('{"latency_ms": 2}\n')
    assert not ev_run._git_sha(repo).endswith("+dirty")

    (repo / "new_module.py").write_text("y = 2\n")
    assert ev_run._git_sha(repo).endswith("+dirty")


def test_rows_from_an_unknown_run_are_not_credited_to_this_one(tiny, tmp_path, fake_embed, monkeypatch):
    fts, hybrid = Variant("fts", mode="fts", index=SMALL), Variant("hybrid", index=SMALL)
    _main(tiny, tmp_path, monkeypatch, [fts, hybrid])
    summary_path = tmp_path / "results" / "tiny" / "summary.json"
    old = json.loads(summary_path.read_text())
    del old["variants"]["hybrid"]["environment"]        # e.g. written by an older harness
    summary_path.write_text(json.dumps(old))

    again = _main(tiny, tmp_path, monkeypatch, [fts])
    assert again["variants"]["hybrid"]["environment"] is None


def test_hits_record_which_chunk_of_the_doc_matched(tmp_path, fake_embed):
    words = [f"w{i}" for i in range(90)]
    words[70] = "zebra"                         # chunks of 50 words step 40: only chunk 1 has it
    ds = Dataset(name="long", docs={"m": " ".join(words)}, queries={"q": "zebra"},
                 qrels={"q": {"m": 1}})
    rows = run_variant(ds, Variant("fts", mode="fts", index=SMALL),
                       build_index(ds, SMALL, tmp_path / "cache"))
    assert rows[0]["hits"] == [
        {"doc": "m", "chunk": 1, "fts_rank": 1, "vec_rank": None, "matched_in": ["fts"]}
    ]


def test_table_renders_before_anything_is_judged(tiny, tmp_path, fake_embed):
    unjudged = Dataset(name="tiny", docs=tiny.docs, queries=tiny.queries, qrels={})
    rows = _run(unjudged, tmp_path, name="hybrid", mode="hybrid")
    table = ev_run.format_table(summarize(unjudged, {"hybrid": rows}, {"hybrid": {}}))
    assert "—" in table.splitlines()[-1]

    empty = Dataset(name="none", docs=tiny.docs, queries={}, qrels={})
    ev_run.format_table(summarize(empty, {"hybrid": []}, {"hybrid": {}}))


def test_rows_record_the_query_text_they_were_run_with(tiny, tmp_path, fake_embed):
    rows = {r["query_id"]: r for r in _run(tiny, tmp_path, mode="fts")}
    assert rows["q1"]["query"] == "water bill"


# ------------------------------------------------------------ phase 4

def test_variant_table_is_the_planned_one():
    from evals.variants import VARIANTS

    qi = "Represent this sentence for searching relevant passages: "
    base = IndexSettings(embed_model="BAAI/bge-base-en-v1.5", embed_dim=768)
    assert VARIANTS == {v.name: v for v in [
        Variant("fts", mode="fts"),
        Variant("vec", mode="vec"),
        Variant("hybrid"),
        Variant("hybrid+qi", query_instruction=qi),
        Variant("hybrid+rerank", rerank=True),
        Variant("vec-base", mode="vec", index=base),
        Variant("hybrid-base", index=base),
        Variant("hybrid-c300", index=IndexSettings(chunk_tokens=300, chunk_overlap=50),
                datasets=("papers",)),
        Variant("hybrid-c200", index=IndexSettings(chunk_tokens=200, chunk_overlap=40),
                datasets=("papers",)),
        Variant("hybrid-k20", rrf_k=20),
        Variant("fts-or", mode="fts", fts_operator="or"),
        Variant("hybrid-or", fts_operator="or"),
    ]}


def test_all_means_every_variant_that_applies_to_the_dataset():
    assert len(resolve("all", "papers")) == 12
    scifact = [v.name for v in resolve("all", "scifact")]
    assert len(scifact) == 10 and "hybrid-c300" not in scifact and "hybrid-c200" not in scifact


def test_a_variant_for_another_dataset_is_refused():
    with pytest.raises(ValueError, match="hybrid-c200.*scifact"):
        resolve("hybrid,hybrid-c200", "scifact")


def _spy_search(monkeypatch):
    calls = []
    real = ev_run.search

    def spy(query, conn, cfg, **kw):
        calls.append((query, kw, cfg.embeddings.query_instruction))
        return real(query, conn, cfg, **kw)

    monkeypatch.setattr(ev_run, "search", spy)
    return calls


def test_variant_options_reach_search(tiny, tmp_path, fake_embed, fake_rerank, monkeypatch):
    calls = _spy_search(monkeypatch)
    _run(tiny, tmp_path, rrf_k=20, fts_operator="or", rerank=True, query_instruction="find: ")
    _, kw, instruction = calls[-1]
    assert (kw["rrf_k"], kw["fts_operator"], kw["rerank"], instruction) == (20, "or", True, "find: ")
    assert kw["limit"] == ev_run.SEARCH_LIMIT


def test_every_query_is_searched_once_before_any_is_timed(tiny, tmp_path, fake_embed, monkeypatch):
    calls = _spy_search(monkeypatch)
    _run(tiny, tmp_path, mode="hybrid")
    queries = list(tiny.queries.values())
    assert [q for q, _, _ in calls] == queries + queries


def test_judged_at_10_is_reported_for_pooled_sets_only(tiny, tmp_path, fake_embed):
    rows = _run(tiny, tmp_path, name="hybrid")
    n_docs = sum(len(r["docs"]) for r in rows)
    graded = sum(d in tiny.qrels.get(r["query_id"], {}) for r in rows for d in r["docs"])
    pooled = Dataset(**{**tiny.__dict__, "pooled": True})
    entry = summarize(pooled, {"hybrid": rows}, {"hybrid": {}})["variants"]["hybrid"]
    assert entry["judged@10"] == pytest.approx(graded / n_docs)
    assert "judged@10" in ev_run.format_table(summarize(pooled, {"hybrid": rows}, {"hybrid": {}}))
    plain = summarize(tiny, {"hybrid": rows}, {"hybrid": {}})
    assert "judged@10" not in plain["variants"]["hybrid"]
    assert "judged@10" not in ev_run.format_table(plain)


def test_summary_records_each_variants_search_settings(tiny, tmp_path, fake_embed, fake_rerank,
                                                       monkeypatch):
    summary = _main(tiny, tmp_path, monkeypatch, [
        Variant("hybrid", index=SMALL),
        Variant("hybrid+rerank", rerank=True, index=SMALL),
    ])
    assert summary["variants"]["hybrid"]["search"] == {
        "rrf_k": 60, "fts_operator": "and", "query_instruction": "", "rerank": None}
    assert summary["variants"]["hybrid+rerank"]["search"]["rerank"] == {
        "model": "BAAI/bge-reranker-base", "revision": "test-rerank-rev", "top_n": 50}
    again = _main(tiny, tmp_path, monkeypatch, [Variant("hybrid", index=SMALL)])
    assert again["variants"]["hybrid+rerank"]["search"] == summary["variants"]["hybrid+rerank"]["search"]


# ------------------------------------------------------------ rescoring

def _main_args(tmp_path, *extra):
    return ["--dataset", "tiny", "--out", str(tmp_path / "results"),
            "--cache", str(tmp_path / "cache"), *extra]


def _no_search(monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("rescoring must not search")
    monkeypatch.setattr(ev_run, "search", refuse)


def test_rescore_rescores_the_rows_on_disk_without_searching(tiny, tmp_path, fake_embed,
                                                             monkeypatch):
    variants = [Variant("fts", mode="fts", index=SMALL), Variant("hybrid", index=SMALL)]
    first = _main(tiny, tmp_path, monkeypatch, variants)
    rows_before = (tmp_path / "results" / "tiny" / "hybrid.jsonl").read_text().splitlines()

    graded = Dataset(**{**tiny.__dict__, "qrels": {**tiny.qrels, "q3": {"a": 2}}})
    monkeypatch.setitem(ev_run.DATASETS, "tiny", lambda cache_dir: graded)
    monkeypatch.setattr(ev_run, "environment", lambda: {"alexandria_commit": "elsewhere"})
    _no_search(monkeypatch)
    ev_run.main(_main_args(tmp_path, "--rescore"))

    again = json.loads((tmp_path / "results" / "tiny" / "summary.json").read_text())
    assert again["variants"]["hybrid"]["metrics"]["ndcg@10"]["n"] == \
        first["variants"]["hybrid"]["metrics"]["ndcg@10"]["n"] + 1      # q3 now counts
    rows_after = (tmp_path / "results" / "tiny" / "hybrid.jsonl").read_text().splitlines()
    before = {json.loads(r)["query_id"]: json.loads(r) for r in rows_before}
    after = {json.loads(r)["query_id"]: json.loads(r) for r in rows_after}
    assert after["q3"]["ndcg@10"] is not None and before["q3"]["ndcg@10"] is None
    for q in after:                     # the search side is untouched
        assert {k: v for k, v in after[q].items() if k not in ev_run.METRICS} == \
               {k: v for k, v in before[q].items() if k not in ev_run.METRICS}
    for name in ("fts", "hybrid"):      # credited to the run that searched, not this one
        assert again["variants"][name]["environment"] == first["variants"][name]["environment"]
        assert again["variants"][name]["latency_ms"] == first["variants"][name]["latency_ms"]


def test_rescore_refuses_rows_for_a_changed_query_set(tiny, tmp_path, fake_embed, monkeypatch):
    _main(tiny, tmp_path, monkeypatch, [Variant("hybrid", index=SMALL)])
    for queries in ({**tiny.queries, "q1": "water bills"},       # reworded
                    {**tiny.queries, "q4": "new question"}):      # added
        changed = Dataset(**{**tiny.__dict__, "queries": queries})
        monkeypatch.setitem(ev_run.DATASETS, "tiny", lambda cache_dir, ds=changed: ds)
        _no_search(monkeypatch)
        with pytest.raises(SystemExit, match="hybrid.*without --rescore"):
            ev_run.main(_main_args(tmp_path, "--rescore"))


def test_rescore_refuses_rows_from_another_index(tiny, tmp_path, fake_embed, monkeypatch):
    _main(tiny, tmp_path, monkeypatch, [Variant("hybrid", index=SMALL)])
    other = IndexSettings(chunk_tokens=40, chunk_overlap=10)
    monkeypatch.setattr(ev_run, "resolve", lambda spec, dataset: [Variant("hybrid", index=other)])
    _no_search(monkeypatch)
    with pytest.raises(SystemExit, match="hybrid.*without --rescore"):
        ev_run.main(_main_args(tmp_path, "--rescore"))


def test_rescore_with_no_rows_on_disk_says_so(tiny, tmp_path, fake_embed, monkeypatch):
    monkeypatch.setitem(ev_run.DATASETS, "tiny", lambda cache_dir: tiny)
    monkeypatch.setattr(ev_run, "resolve", lambda spec, dataset: [Variant("hybrid", index=SMALL)])
    _no_search(monkeypatch)
    with pytest.raises(SystemExit, match="no rows to rescore"):
        ev_run.main(_main_args(tmp_path, "--rescore"))
