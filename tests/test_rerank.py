"""The opt-in cross-encoder reranker: config, search, CLI and MCP wiring.
Uses the fake embedder and the fake cross-encoder (conftest)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from typer.testing import CliRunner

import alexandria.cli as cli
import alexandria.server as srv
from alexandria.config import SearchConfig
from alexandria.config import load as load_config
from alexandria.ingest import ingest_file
from alexandria.search import search

DOCS = {
    "a.txt": "Rerankers score each query and passage pair with a cross-encoder.",
    "b.txt": "A passage about gardening, soil and tomatoes in summer.",
    "c.txt": "Query expansion adds words; a reranker reads the query and the passage together.",
    "d.txt": "Fusion of keyword and vector rankings with reciprocal ranks.",
    "e.txt": "Notes on the passage of time and the query of a curious cat.",
}


@pytest.fixture
def seeded(cfg, conn, tmp_path: Path, fake_embed):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in DOCS.items():
        (folder / name).write_text(text)
        ingest_file(folder / name, conn, cfg, category="notes" if name < "c" else "other")
    return cfg, conn


def _with(cfg, **search_fields):
    return dataclasses.replace(cfg, search=dataclasses.replace(cfg.search, **search_fields))


def test_search_config_defaults(cfg):
    assert cfg.search == SearchConfig(rerank=False, rerank_model="BAAI/bge-reranker-base",
                                      rerank_top_n=50)


def test_search_config_reads_the_search_table(home, monkeypatch):
    (home / "config.toml").write_text(
        '[search]\nrerank = true\nrerank_model = "some/cross-encoder"\nrerank_top_n = 7\n'
    )
    monkeypatch.setenv("ALEXANDRIA_HOME", str(home))
    assert load_config().search == SearchConfig(True, "some/cross-encoder", 7)


def test_rerank_off_never_calls_the_cross_encoder(seeded, fake_rerank):
    cfg, conn = seeded
    hits = search("query passage", conn, cfg, limit=5)
    assert fake_rerank == []
    assert all(h.rerank_score is None for h in hits)
    assert hits == search("query passage", conn, cfg, limit=5, rerank=False)


def test_rerank_reorders_the_top_n_fused_chunks(seeded, fake_rerank):
    cfg, conn = _with(seeded[0], rerank_top_n=3), seeded[1]
    fused = search("query passage", conn, cfg, limit=5)
    hits = search("query passage", conn, cfg, limit=5, rerank=True)

    [(query, texts)] = fake_rerank
    assert query == "query passage"
    chunk_text = {cid: t for cid, t in conn.execute("SELECT id, text FROM chunks")}
    assert texts == [chunk_text[h.chunk_id] for h in fused[:3]]     # in fused order

    head, tail = hits[:3], hits[3:]
    assert {h.chunk_id for h in head} == {h.chunk_id for h in fused[:3]}
    scores = [h.rerank_score for h in head]
    assert None not in scores and scores == sorted(scores, reverse=True)
    fused_pos = {h.chunk_id: i for i, h in enumerate(fused)}
    for x, y in zip(head, head[1:]):                       # ties keep fused order
        if x.rerank_score == y.rerank_score:
            assert fused_pos[x.chunk_id] < fused_pos[y.chunk_id]
    assert [h.chunk_id for h in tail] == [h.chunk_id for h in fused[3:]]
    assert all(h.rerank_score is None for h in tail)
    by_id = {h.chunk_id: h for h in fused}
    for h in hits:                                         # retriever ranks kept
        assert (h.fts_rank, h.vec_rank, h.matched_in) == (
            by_id[h.chunk_id].fts_rank, by_id[h.chunk_id].vec_rank, by_id[h.chunk_id].matched_in)


def test_rerank_puts_the_best_scored_chunk_first(seeded, fake_rerank):
    cfg, conn = seeded
    hits = search("query passage", conn, cfg, limit=5, rerank=True)
    assert Path(hits[0].source_uri).name in {"c.txt", "e.txt"}   # most query words
    assert hits[0].rerank_score == max(h.rerank_score for h in hits)


def test_limit_applies_after_reranking_and_only_top_n_are_scored(seeded, fake_rerank):
    cfg, conn = _with(seeded[0], rerank_top_n=2), seeded[1]
    hits = search("query passage", conn, cfg, limit=4, rerank=True)
    assert len(fake_rerank[0][1]) == 2
    assert len(hits) == 4


def test_rerank_respects_the_category_filter(seeded, fake_rerank):
    cfg, conn = seeded
    hits = search("query passage", conn, cfg, category="notes", rerank=True)
    assert hits and {h.category for h in hits} == {"notes"}
    assert len(fake_rerank[0][1]) == len(hits)


def test_rerank_with_no_candidates_returns_nothing(seeded, fake_rerank):
    cfg, conn = seeded
    assert search("query passage", conn, cfg, category="nope", rerank=True) == []
    assert fake_rerank == []


# ------------------------------------------------------------------ wiring

@pytest.fixture
def spy_search(monkeypatch):
    calls: list[dict] = []

    def _spy(query, conn, cfg, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(cli, "search", _spy)
    monkeypatch.setattr(srv, "search", _spy)
    return calls


@pytest.mark.parametrize("config, args, expected", [
    ("", [], False),
    ("", ["--rerank"], True),
    ("[search]\nrerank = true\n", [], True),
    ("[search]\nrerank = true\n", ["--no-rerank"], False),
])
def test_cli_rerank_flag_and_config_default(cfg, home, spy_search, config, args, expected):
    (home / "config.toml").write_text(config)
    result = CliRunner().invoke(cli.app, ["search", "some query", *args])
    assert result.exit_code == 0, result.output
    assert spy_search[0]["rerank"] is expected


def test_mcp_search_never_reranks_even_when_config_says_so(cfg, conn, monkeypatch, spy_search):
    monkeypatch.setattr(srv, "_get", lambda: (_with(cfg, rerank=True), conn))
    srv.search_tool("some query")
    assert spy_search[0].get("rerank", False) is False
