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
        {"doc": "c", "fts_rank": 1, "vec_rank": None, "matched_in": ["fts"]}
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
    assert [v.name for v in resolve("all")] == ["fts", "vec", "hybrid"]
    assert [v.name for v in resolve("vec, fts")] == ["vec", "fts"]
    with pytest.raises(ValueError, match="nope"):
        resolve("fts,nope")


def test_main_writes_rows_and_summary(tiny, tmp_path, fake_embed, monkeypatch):
    monkeypatch.setitem(ev_run.DATASETS, "tiny", lambda cache_dir: tiny)
    monkeypatch.setattr(ev_run, "resolve", lambda spec: [
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
    monkeypatch.setattr(ev_run, "resolve", lambda spec: [Variant("fts", mode="fts", index=SMALL)])
    out = tmp_path / "results"
    ev_run.main(["--dataset", "tiny", "--out", str(out), "--cache", str(tmp_path / "cache")])
    summary = json.loads((out / "tiny" / "summary.json").read_text())
    assert summary["variants"]["fts"]["environment"] == {"alexandria_commit": "start"}


def _main(tiny, tmp_path, monkeypatch, variants):
    monkeypatch.setitem(ev_run.DATASETS, "tiny", lambda cache_dir: tiny)
    monkeypatch.setattr(ev_run, "resolve", lambda spec: variants)
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
