"""Token counts of indexed chunks against the embedding model's window."""
import json

import pytest

from evals import run as ev_run
from evals import tokens as ev_tokens
from evals.datasets import Dataset
from evals.index import IndexSettings
from evals.tokens import chunk_settings, measure
from evals.variants import Variant


def count_words(texts, special):
    """A tokenizer that makes each word one token, plus [CLS] and [SEP]."""
    return [len(t.split()) + (2 if special else 0) for t in texts]


def test_tokens_per_word_is_over_all_the_text_not_a_mean_of_ratios():
    def two_per_word_then_one(texts, special):
        return [2 * len(t.split()) if i == 0 else len(t.split()) for i, t in enumerate(texts)]

    out = measure(["a b c d", "e"], {}, two_per_word_then_one, window=512)
    assert out["tokens_per_word"] == pytest.approx(9 / 5)


def test_a_chunk_overflows_only_past_the_window_with_its_special_tokens():
    # 2, 3 and 4 words: 4, 5 and 6 tokens with [CLS] and [SEP].
    out = measure([], {"800/100": ["a b", "a b c", "a b c d"]}, count_words, window=4)
    sizes = out["chunks"]["800/100"]
    assert sizes["n"] == 3
    assert sizes["over_window"] == pytest.approx(2 / 3)    # exactly the window still fits
    assert sizes["median"] == 5
    assert sizes["p95"] == pytest.approx(5.9)


def test_documents_over_the_window_are_counted_whole():
    out = measure(["a b", "a b c d e"], {}, count_words, window=5)
    assert out["docs"] == {"n": 2, "over_window": 0.5}


def test_every_chunk_size_the_datasets_variants_use_is_measured_once():
    assert [(s.chunk_tokens, s.chunk_overlap) for s in chunk_settings("papers")] == [
        (800, 100), (300, 50), (200, 40),
    ]
    assert [(s.chunk_tokens, s.chunk_overlap) for s in chunk_settings("scifact")] == [(800, 100)]
    # Measured with the default model's tokenizer; bge-base's is the same.
    assert {s.embed_model for s in chunk_settings("papers")} == {IndexSettings().embed_model}


class FakeTokenizer:
    def __call__(self, texts, add_special_tokens, verbose):
        return {"input_ids": [[0] * n for n in count_words(texts, add_special_tokens)]}


class FakeModel:
    tokenizer = FakeTokenizer()
    max_seq_length = 6


def test_main_writes_the_counts_for_each_chunk_size(tmp_path, fake_embed, monkeypatch):
    words = " ".join(f"w{i}" for i in range(12))
    tiny = Dataset(name="tiny", docs={"a": words, "b": "short text"},
                   queries={"q1": "short"}, qrels={"q1": {"b": 1}})
    monkeypatch.setitem(ev_run.DATASETS, "tiny", lambda cache_dir: tiny)
    monkeypatch.setattr(ev_tokens, "resolve", lambda spec, dataset: [
        Variant("hybrid", index=IndexSettings(chunk_tokens=8, chunk_overlap=2)),
        Variant("hybrid-small", index=IndexSettings(chunk_tokens=4, chunk_overlap=1)),
    ])
    monkeypatch.setattr(ev_tokens, "_get_model", lambda name, device: FakeModel())
    monkeypatch.setattr("evals.index.model_revision", lambda name: "test-rev")

    out = tmp_path / "results"
    ev_tokens.main(["--dataset", "tiny", "--out", str(out), "--cache", str(tmp_path / "cache")])

    got = json.loads((out / "tiny" / "tokens.json").read_text())
    assert got["window"] == 6
    assert got["tokens_per_word"] == pytest.approx(1.0)
    assert got["docs"] == {"n": 2, "over_window": 0.5}
    assert set(got["chunks"]) == {"8/2", "4/1"}
    # 12 words in 8-word chunks with 2 overlapping: 8 + 6 words, 10 and 8 tokens.
    assert got["chunks"]["8/2"]["n"] == 3                  # with "short text"
    assert got["chunks"]["8/2"]["over_window"] == pytest.approx(2 / 3)
    assert got["chunks"]["4/1"]["over_window"] == 0.0
