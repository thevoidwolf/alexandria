"""BEIR SciFact (CC BY-SA 4.0): ~5k paper abstracts, 300 test claims.

Downloaded from the Hugging Face Hub at pinned revisions into the eval cache;
nothing from it is committed.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from evals.datasets import Dataset

CORPUS_REPO = "BeIR/scifact"
CORPUS_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
QRELS_REPO = "BeIR/scifact-qrels"
QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"


def build(
    corpus_rows: Iterable[Mapping[str, str]],
    query_rows: Iterable[Mapping[str, str]],
    qrels_tsv: str,
) -> Dataset:
    """Assemble the test split. Bodies are ``title + "\\n\\n" + text``."""
    qrels: dict[str, dict[str, int]] = {}
    for line in qrels_tsv.splitlines()[1:]:  # skip the header
        if not line.strip():
            continue
        qid, doc_id, grade = line.split("\t")
        qrels.setdefault(qid, {})[doc_id] = int(grade)

    docs = {
        r["_id"]: f"{r['title']}\n\n{r['text']}" if r["title"].strip() else r["text"]
        for r in corpus_rows
    }
    queries = {r["_id"]: r["text"] for r in query_rows if r["_id"] in qrels}
    return Dataset(
        name="scifact",
        docs=docs,
        queries=queries,
        qrels=qrels,
        source={CORPUS_REPO: CORPUS_REVISION, QRELS_REPO: QRELS_REVISION},
    )


def load(cache_dir: Path) -> Dataset:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def fetch(repo: str, revision: str, filename: str) -> Path:
        return Path(hf_hub_download(
            repo, filename, repo_type="dataset", revision=revision,
            cache_dir=cache_dir / "hf",
        ))

    corpus = pq.read_table(
        fetch(CORPUS_REPO, CORPUS_REVISION, "corpus/corpus-00000-of-00001.parquet")
    ).to_pylist()
    queries = pq.read_table(
        fetch(CORPUS_REPO, CORPUS_REVISION, "queries/queries-00000-of-00001.parquet")
    ).to_pylist()
    qrels = fetch(QRELS_REPO, QRELS_REVISION, "test.tsv").read_text()
    return build(corpus, queries, qrels)
