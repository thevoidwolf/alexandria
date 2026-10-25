"""How much of each indexed chunk the embedding model can read.

    uv run python -m evals.tokens --dataset papers

Chunks are sized in words, but bge-small-en-v1.5 reads at most 512 of its
own tokens and ignores the rest. For each chunk size the dataset's variants
use, this counts every chunk in the built index with the model's tokenizer,
[CLS] and [SEP] included, as the model sees them, and the documents'
extracted text without them. It writes ``<out>/<dataset>/tokens.json``:
tokens per word, the share of documents and of chunks longer than the
window, and the median and 95th-percentile chunk length. bge-base-en-v1.5
has the same tokenizer and window, so its variants' chunks count the same.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from alexandria.config import EmbeddingsConfig
from alexandria.ingest.embed import _get_model
from evals.index import IndexSettings, build_index, open_index
from evals.run import DATASETS, REPO
from evals.variants import resolve

# (texts, with [CLS]/[SEP]?) -> token count of each text
Counter = Callable[[Sequence[str], bool], list[int]]


def _size(s: IndexSettings) -> str:
    return f"{s.chunk_tokens}/{s.chunk_overlap}"


def chunk_settings(dataset: str) -> list[IndexSettings]:
    """One index per chunk size the dataset's variants use, in variant order."""
    sizes: dict[str, IndexSettings] = {}
    for v in resolve("all", dataset):
        sizes.setdefault(_size(v.index), replace(
            IndexSettings(), chunk_tokens=v.index.chunk_tokens, chunk_overlap=v.index.chunk_overlap,
        ))
    return list(sizes.values())


def measure(
    doc_texts: Sequence[str], chunks: Mapping[str, Sequence[str]], count: Counter, window: int,
) -> dict:
    """``chunks`` maps a chunk size ("800/100") to the text of every chunk."""
    words = sum(len(t.split()) for t in doc_texts)
    doc_lengths = count(doc_texts, True)
    out = {
        "window": window,
        "tokens_per_word": sum(count(doc_texts, False)) / words if words else None,
        "docs": {
            "n": len(doc_texts),
            "over_window": sum(n > window for n in doc_lengths) / len(doc_texts) if doc_texts else None,
        },
        "chunks": {},
    }
    for size, texts in chunks.items():
        lengths = np.asarray(count(texts, True))
        out["chunks"][size] = {
            "n": len(texts),
            "over_window": float((lengths > window).mean()),
            "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m evals.tokens", description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--out", type=Path, default=REPO / "evals" / "results")
    p.add_argument("--cache", type=Path, default=REPO / "evals" / ".cache")
    args = p.parse_args(argv)

    ds = DATASETS[args.dataset](args.cache)
    settings = chunk_settings(ds.name)
    model = _get_model(settings[0].embed_model, EmbeddingsConfig().device)

    def count(texts: Sequence[str], special: bool) -> list[int]:
        ids = model.tokenizer(list(texts), add_special_tokens=special, verbose=False)["input_ids"]
        return [len(i) for i in ids]

    doc_texts: list[str] = []
    chunks: dict[str, list[str]] = {}
    for s in settings:
        _, conn = open_index(build_index(ds, s, args.cache))
        try:
            if not doc_texts:   # extraction doesn't depend on the chunk size
                doc_texts = [r[0] for r in conn.execute("SELECT extracted_text FROM documents")]
            chunks[_size(s)] = [r[0] for r in conn.execute("SELECT text FROM chunks")]
        finally:
            conn.close()

    result = {"dataset": ds.name, "model": settings[0].embed_model,
              **measure(doc_texts, chunks, count, model.max_seq_length)}
    out_dir = args.out / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokens.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"{ds.name}: {result['tokens_per_word']:.2f} tokens per word; "
          f"{result['docs']['over_window']:.1%} of documents over {result['window']} tokens")
    for size, c in result["chunks"].items():
        print(f"  {size:>8} words: {c['n']:>5} chunks, {c['over_window']:.1%} over the window, "
              f"median {c['median']:.0f}, p95 {c['p95']:.0f}")


if __name__ == "__main__":
    main()
