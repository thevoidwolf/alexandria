"""Blind pooled judging for the papers set.

    uv run python -m evals.judge

For each query, the top 10 documents from every variant that has been run
are pooled and shuffled, and each unjudged (query, paper) pair is shown with
the paper's title and a stretch of a retrieved passage. Nothing says which
variant found a paper, at what rank, or how many variants found it, and
passages are trimmed to a fixed length so chunk size can't give a variant
away either. Grades go straight into
``evals/data/papers/qrels.tsv``, one line per judgement, so quitting at any
point loses nothing and the next session picks up where this one stopped.

    0  not relevant
    1  relevant: the paper addresses the query
    2  highly relevant: the paper is what the query is looking for

Papers no variant returned are never judged and count as not relevant (the
usual bias of pooled judgements). Running a new variant later adds its
unjudged pairs to the pool.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evals.datasets import Dataset, papers
from evals.index import BuiltIndex, build_index, index_key, open_index
from evals.variants import VARIANTS

REPO = Path(__file__).resolve().parent.parent
GRADES = {"0": 0, "1": 1, "2": 2}
PROMPT = "grade 0 / 1 / 2, m = more of the passage, s = skip, q = quit > "
WINDOW_WORDS = 80
# The longer view is capped at the smallest chunk any variant uses (200
# words), so a passage's length can't reveal a chunk-size variant.
MORE_WORDS = 200
_STOPWORDS = frozenset(
    "a an and are do does for how in is of on the to what which with".split()
)


@dataclass(frozen=True)
class Pair:
    query_id: str
    doc: str
    # (variant, chunk ord) of every retrieval of this doc for this query,
    # best rank first. Used only to fetch passages; never shown.
    chunks: tuple[tuple[str, int], ...]


def pool(
    rows_by_variant: Mapping[str, list[dict]],
    qrels: Mapping[str, Mapping[str, int]],
    query_order: Sequence[str],
    seed: int = 0,
) -> list[Pair]:
    """Unjudged (query, doc) pairs from every variant's top 10, queries in
    order, docs within a query in a shuffled order fixed by ``seed``."""
    found: dict[tuple[str, str], list[tuple[int, str, int]]] = defaultdict(list)
    for variant, rows in rows_by_variant.items():
        for r in rows:
            for rank, hit in enumerate(r["hits"]):
                found[(r["query_id"], hit["doc"])].append((rank, variant, hit["chunk"]))
    by_query: dict[str, list[str]] = defaultdict(list)
    for qid, doc in found:
        if doc not in qrels.get(qid, {}):
            by_query[qid].append(doc)

    pairs = []
    for qid in query_order:
        docs = sorted(by_query.get(qid, []))
        random.Random(f"{seed}:{qid}").shuffle(docs)
        for doc in docs:
            seen = sorted(found[(qid, doc)])
            chunks = tuple(dict.fromkeys((variant, ord_) for _, variant, ord_ in seen))
            pairs.append(Pair(qid, doc, chunks))
    return pairs


def _tokens(s: str) -> list[str]:
    return re.findall(r"[^\W_]+", s.casefold())


def _best_span(words: list[str], query: str, width: int) -> tuple[int, int]:
    """(query words in it, start) of the best ``width``-word stretch."""
    wanted = {t for t in _tokens(query) if t not in _STOPWORDS}
    hits = [any(t in wanted for t in _tokens(w)) for w in words]
    best = score = sum(hits[:width])
    start = 0
    for i in range(1, len(words) - width + 1):
        score += hits[i + width - 1] - hits[i - 1]
        if score > best:
            best, start = score, i
    return best, start


def best_window(text: str, query: str, width: int = WINDOW_WORDS) -> str:
    """The ``width``-word stretch of ``text`` with the most query words."""
    words = text.split()
    if len(words) <= width:
        return " ".join(words)
    _, start = _best_span(words, query, width)
    end = start + width
    return ("… " if start else "") + " ".join(words[start:end]) + (" …" if end < len(words) else "")


def judge(
    pairs: Sequence[Pair],
    *,
    query_text: Mapping[str, str],
    title: Mapping[str, str],
    chunk_text: Callable[[str, str, int], str],
    ask: Callable[[str], str],
    show: Callable[[str], None],
    record: Callable[[Pair, int], None],
    window: int = WINDOW_WORDS,
    more: int = MORE_WORDS,
) -> int:
    """Ask for a grade for each pair; returns how many were graded.

    One passage is shown per pair: of the passages the variants retrieved,
    the one whose best stretch has the most query words (ties go to the
    best-ranked). How many there were is never shown.
    """
    graded = 0
    for k, pair in enumerate(pairs, 1):
        query = query_text[pair.query_id]
        passages = list(dict.fromkeys(chunk_text(v, pair.doc, o) for v, o in pair.chunks))
        passage = max(passages, key=lambda t: _best_span(t.split(), query, window)[0])
        show(
            f"\n── {k} of {len(pairs)} " + "─" * 40
            + f"\nQuery:  {query}"
            + f"\nPaper:  {title.get(pair.doc, pair.doc)}  [{pair.doc}]"
            + f"\n\n{best_window(passage, query, window)}\n"
        )
        while True:
            try:
                key = ask(PROMPT).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return graded
            if key in GRADES:
                record(pair, GRADES[key])
                graded += 1
                break
            if key == "m":
                show(f"\n{best_window(passage, query, more)}\n")
            elif key == "s":
                break
            elif key == "q":
                return graded
            else:
                show("  (0, 1, 2, m, s or q)")
    return graded


def append_grade(path: Path, query_id: str, doc: str, grade: int) -> None:
    """Add one judgement to qrels.tsv, on its own line even if the file was
    hand-edited and lost its final newline."""
    sep = ""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            sep = "" if f.read(1) == b"\n" else "\n"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"{sep}{query_id}\t{doc}\t{grade}\n")


def session(
    ds: Dataset,
    rows_by_variant: Mapping[str, list[dict]],
    builds: Mapping[str, BuiltIndex],
    qrels_path: Path,
    *,
    ask: Callable[[str], str],
    show: Callable[[str], None],
    seed: int = 0,
) -> int:
    """Judge every unjudged pooled pair, appending each grade to ``qrels_path``."""
    pairs = pool(rows_by_variant, ds.qrels, list(ds.queries), seed)
    show(f"{len(pairs)} unjudged (query, paper) pairs across "
         f"{len({p.query_id for p in pairs})} queries.")
    conns = {v: open_index(b)[1] for v, b in builds.items()}

    def chunk_text(variant: str, doc: str, ord_: int) -> str:
        row = conns[variant].execute(
            "SELECT text FROM chunks WHERE doc_id = ? AND ord = ?",
            (builds[variant].doc_map[doc], ord_),
        ).fetchone()
        return row[0]

    def record(pair: Pair, grade: int) -> None:
        append_grade(qrels_path, pair.query_id, pair.doc, grade)

    try:
        return judge(pairs, query_text=ds.queries, title=ds.titles, chunk_text=chunk_text,
                     ask=ask, show=show, record=record)
    finally:
        for c in conns.values():
            c.close()


def check_fresh(summary_variants: Mapping[str, dict], keys: Mapping[str, str]) -> None:
    """Refuse rows produced from a different index than the current one:
    their chunk positions would point at the wrong passages."""
    for variant, key in keys.items():
        ran_on = summary_variants.get(variant, {}).get("index", {}).get("key")
        if ran_on != key:
            raise ValueError(
                f"{variant}'s results came from index {ran_on}, but the corpus, settings or "
                f"ingest code now give {key}: re-run `uv run python -m evals.run --dataset "
                f"papers --variants {variant}` before judging"
            )


def check_queries(rows_by_variant: Mapping[str, list[dict]], queries: Mapping[str, str]) -> None:
    """Refuse rows run with a query's earlier wording: pooling them would show
    results for one question under another."""
    stale = sorted({
        r["query_id"] for rows in rows_by_variant.values() for r in rows
        if r["query_id"] in queries and r.get("query") != queries[r["query_id"]]
    })
    if stale:
        raise ValueError(
            f"queries reworded since their results were produced: {', '.join(stale)}. Re-run "
            "`uv run python -m evals.run --dataset papers --variants all` before judging "
            "(and give a reworded query a new id if it was already judged)"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m evals.judge", description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, default=REPO / "evals" / "results")
    p.add_argument("--cache", type=Path, default=REPO / "evals" / ".cache")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    ds = papers.load(args.cache)
    res_dir = args.results / ds.name
    summary_path = res_dir / "summary.json"
    rows = {
        path.stem: [json.loads(line) for line in path.read_text().splitlines()]
        for path in sorted(res_dir.glob("*.jsonl")) if path.stem in VARIANTS
    }
    if not rows or not summary_path.exists():
        sys.exit("No papers results to pool yet: run "
                 "`uv run python -m evals.run --dataset papers --variants all` first.")
    try:
        check_queries(rows, ds.queries)
        check_fresh(json.loads(summary_path.read_text())["variants"],
                    {v: index_key(ds, VARIANTS[v].index) for v in rows})
    except ValueError as e:
        sys.exit(str(e))
    builds = {v: build_index(ds, VARIANTS[v].index, args.cache) for v in rows}
    n = session(ds, rows, builds, papers.DATA_DIR / "qrels.tsv", ask=input, show=print,
                seed=args.seed)
    print(f"\n{n} judgement(s) added to {papers.DATA_DIR / 'qrels.tsv'}. "
          "Score them with `uv run python -m evals.run --dataset papers --rescore`.")


if __name__ == "__main__":
    main()
