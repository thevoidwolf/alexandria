"""Results -> the generated tables and charts in REPORT.md.

    uv run python -m evals.report

REPORT.md is written by hand, apart from blocks like

    <!-- report:main papers -->
    (generated)
    <!-- /report -->

which this fills from ``evals/results/`` and each dataset's queries and
judgements, leaving everything outside them as it is. Charts go to
``evals/results/charts/``.

On a pooled set (papers), a variant's ungraded papers count as not
relevant, and every new grade can move every variant's scores: a relevant
paper one variant found raises the ideal ranking for all of them. So blocks
built on grades are left untouched until every variant's top 10 is judged
(``judged@10`` 100%), and the run lists what it skipped. ``--provisional``
renders them anyway, marked as such, for a look before judging is done.
Don't commit that.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evals.metrics import paired_bootstrap
from evals.run import BOOTSTRAP, DATASETS, REPO
from evals.variants import VARIANTS

LENGTHS = (("1–4", 1, 4), ("5–8", 5, 8), ("9–12", 9, 12), ("13+", 13, None))
RANK_BANDS = (("1–3", 0, 3), ("4–10", 3, 10))
BUCKETS = (("both", ("fts", "vec")), ("fts", ("fts",)), ("vec", ("vec",)))
TYPES = ("lookup", "paraphrase", "conceptual")
MIN_HITS = 10           # fewest hits of each kind a precision gap is reported for


@dataclass
class Results:
    name: str
    summary: dict
    rows: dict[str, dict[str, dict]]        # variant -> query id -> row
    queries: dict[str, str]
    query_meta: dict[str, dict]
    qrels: dict[str, dict[str, int]]
    pooled: bool
    tokens: dict | None

    @property
    def variants(self) -> list[str]:
        """Variants with rows, in the variant table's order."""
        return [v for v in VARIANTS if v in self.rows]

    @property
    def complete(self) -> bool:
        """Every variant's top 10 has a grade (always so for a non-pooled set)."""
        return not self.pooled or all(
            v.get("judged@10") == 1.0 for v in self.summary["variants"].values()
        )


def load(name: str, results_dir: Path, cache: Path) -> Results:
    ds = DATASETS[name](cache)
    res = results_dir / ds.name
    summary = json.loads((res / "summary.json").read_text())
    collapsed = [v for v, e in summary["variants"].items() if e["index"].get("n_collapsed")]
    if collapsed:
        # Grades are read by dataset id below; a collapsed document would need translating.
        raise SystemExit(f"{name}: {', '.join(collapsed)} collapsed duplicate documents; "
                         "the report doesn't handle that yet")
    rows = {
        p.stem: {r["query_id"]: r for r in map(json.loads, p.read_text().splitlines())}
        for p in sorted(res.glob("*.jsonl")) if p.stem in VARIANTS
    }
    tokens = res / "tokens.json"
    return Results(
        name=ds.name, summary=summary, rows=rows, queries=ds.queries, query_meta=ds.query_meta,
        qrels=ds.qrels, pooled=ds.pooled,
        tokens=json.loads(tokens.read_text()) if tokens.exists() else None,
    )


# --- numbers ---------------------------------------------------------------

def _num(x: float | None, spec: str = ".4f") -> str:
    return "—" if x is None else format(x, spec).replace("-", "−")


def _signed(x: float) -> str:
    return _num(x, "+.3f")


def _ci(diff: float, lo: float, hi: float) -> str:
    text = f"{_signed(diff)} [{_signed(lo)}, {_signed(hi)}]"
    return f"**{text}**" if lo > 0 or hi < 0 else text


def _ms(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}" if x < 100 else f"{x:,.0f}"


def _share(k: int, n: int) -> str:
    return f"{k} of {n} ({k / n:.0%})" if n else "—"


def _defined(rows: Mapping[str, dict], metric: str = "ndcg@10") -> list[str]:
    return [q for q, r in rows.items() if r[metric] is not None]


def compare(base: Mapping[str, dict], other: Mapping[str, dict], qids=None,
            metric: str = "ndcg@10") -> tuple[float, float, float, int] | None:
    """Paired bootstrap of ``other - base`` over the queries both define."""
    qids = [q for q in (qids if qids is not None else base)
            if q in other and base[q][metric] is not None and other[q][metric] is not None]
    if not qids:
        return None
    diff, lo, hi = paired_bootstrap([base[q][metric] for q in qids],
                                    [other[q][metric] for q in qids], **BOOTSTRAP)
    return diff, lo, hi, len(qids)


def _mean(rows: Mapping[str, dict], qids=None, metric: str = "ndcg@10") -> float | None:
    vals = [rows[q][metric] for q in (qids if qids is not None else rows)
            if rows[q][metric] is not None]
    return float(np.mean(vals)) if vals else None


def by_length(rows: Mapping[str, dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {label: [] for label, _, _ in LENGTHS}
    for r in rows.values():
        for label, lo, hi in LENGTHS:
            if r["n_words"] >= lo and (hi is None or r["n_words"] <= hi):
                out[label].append(r)
    return out


def fallback_rows(hybrid: Mapping[str, dict], hybrid_or: Mapping[str, dict]) -> dict[str, dict]:
    """AND first, OR only when AND finds nothing (REPORT's decision rule):
    ``hybrid-or``'s row where ``hybrid`` had no FTS hits, ``hybrid``'s otherwise."""
    return {q: hybrid_or[q] if r["n_fts"] == 0 else r for q, r in hybrid.items()}


def matched_in_precision(
    rows: Mapping[str, dict], qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, tuple[int, int]]]:
    """bucket -> rank band ("all", "1–3", "4–10") -> (papers, relevant ones).
    Each returned paper counts once, by the retrievers that found its best
    chunk; unjudged papers count as not relevant."""
    out = {b: {band: [0, 0] for band in ("all", *(lbl for lbl, _, _ in RANK_BANDS))}
           for b, _ in BUCKETS}
    which = {m: b for b, m in BUCKETS}
    for qid, r in rows.items():
        for i, h in enumerate(r["hits"]):
            b = which[tuple(h["matched_in"])]
            rel = qrels.get(qid, {}).get(h["doc"], 0) >= 1
            for band, lo, hi in (("all", 0, None), *RANK_BANDS):
                if i >= lo and (hi is None or i < hi):
                    out[b][band][0] += 1
                    out[b][band][1] += rel
    return {b: {band: tuple(v) for band, v in bands.items()} for b, bands in out.items()}


def precision_gap(
    rows: Mapping[str, dict], qrels: Mapping[str, Mapping[str, int]], lo: int, hi: int,
) -> tuple[float, float, float] | None:
    """Precision of documents both retrievers found minus that of documents
    one found, at ranks ``lo + 1`` to ``hi``, with a 95% CI from resampling
    queries (hits from one query aren't independent). None when either kind
    has fewer than MIN_HITS hits: a handful, all irrelevant, would give a
    falsely narrow interval."""
    counts = []     # per query: both hits, relevant ones, single hits, relevant ones
    for qid, r in rows.items():
        c = [0, 0, 0, 0]
        for h in r["hits"][lo:hi]:
            k = 0 if len(h["matched_in"]) == 2 else 2
            c[k] += 1
            c[k + 1] += qrels.get(qid, {}).get(h["doc"], 0) >= 1
        counts.append(c)
    a = np.asarray(counts)
    if not len(a) or a[:, 0].sum() < MIN_HITS or a[:, 2].sum() < MIN_HITS:
        return None
    point = a[:, 1].sum() / a[:, 0].sum() - a[:, 3].sum() / a[:, 2].sum()
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    s = a[rng.integers(0, len(a), size=(BOOTSTRAP["n_resamples"], len(a)))].sum(axis=1)
    s = s[(s[:, 0] > 0) & (s[:, 2] > 0)]   # resamples that drew both kinds
    gaps = s[:, 1] / s[:, 0] - s[:, 3] / s[:, 2]
    alpha = BOOTSTRAP["alpha"]
    ci_lo, ci_hi = np.quantile(gaps, [alpha / 2, 1 - alpha / 2])
    return float(point), float(ci_lo), float(ci_hi)


def top_hit_score(r: dict, rrf_k: int) -> float | None:
    """The fused score search() gave the top chunk, from its retriever ranks."""
    if not r["hits"]:
        return None
    h = r["hits"][0]
    return sum(1 / (rrf_k + rank) for rank in (h["fts_rank"], h["vec_rank"]) if rank is not None)


# --- blocks ----------------------------------------------------------------

def _table(head: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines)


def main_table(r: Results) -> str:
    base = r.summary["baseline"]
    body = []
    for name in r.variants:
        v = r.summary["variants"][name]
        m = {k: v["metrics"][k]["mean"] for k in ("ndcg@10", "recall@5", "recall@10", "mrr@10")}
        d = v.get("vs_baseline", {}).get("ndcg@10")
        delta = "baseline" if name == base else _ci(d["diff"], d["lo"], d["hi"]) if d else "—"
        lat = v["latency_ms"]
        body.append([f"`{name}`", *(_num(x) for x in m.values()), delta,
                     f"{_ms(lat['p50'])} / {_ms(lat['p95'])}"])
    n = r.summary["variants"][base]["metrics"]["ndcg@10"]["n"]
    note = (f"\n\nMeans over the {n} of {len(r.queries)} queries with at least one relevant "
            "document; the others have no score." if n < len(r.queries) else "")
    return _table(["Variant", "nDCG@10", "Recall@5", "Recall@10", "MRR@10",
                   "Δ nDCG@10 [95% CI]", "p50 / p95 ms"], body) + note


def cost_table(r: Results) -> str:
    by_key: dict[str, list[str]] = {}
    for name in r.variants:
        by_key.setdefault(r.summary["variants"][name]["index"]["key"], []).append(name)
    body = []
    for names in by_key.values():
        idx = r.summary["variants"][names[0]]["index"]
        s = idx["settings"]
        model = s["embed_model"].split("/")[-1].removesuffix("-en-v1.5")
        body.append([f"{model}, {s['chunk_tokens']}/{s['chunk_overlap']} words",
                     ", ".join(f"`{n}`" for n in names), f"{idx['n_docs']:,}", f"{idx['n_chunks']:,}",
                     f"{idx['build_seconds']:.0f}", f"{idx['db_bytes'] / 1e6:.1f}"])
    cpus = sorted({r.summary["variants"][n]["environment"]["cpu"] for n in r.variants})
    return (_table(["Index (model, chunk/overlap)", "Variants", "Documents", "Chunks",
                    "Build (s)", "Database (MB)"], body)
            + f"\n\nBuilt on {', '.join(cpus)}, embedding on CPU. A build is timed once, "
              "and build times move with machine load.")


def tokens_table(r: Results) -> str:
    t = r.tokens
    if t is None:
        raise SystemExit(f"no tokens.json for {r.name}: run `uv run python -m evals.tokens "
                         f"--dataset {r.name}` first")
    sizes = sorted(t["chunks"].items(), key=lambda kv: -int(kv[0].split("/")[0]))
    body = [[size, f"{c['n']:,}", f"{c['over_window']:.1%}", f"{c['median']:,.0f}",
             f"{c['p95']:,.0f}"] for size, c in sizes]
    over, n = t["docs"]["over_window"], t["docs"]["n"]
    docs = f"all {n:,} documents are" if over == 1 else f"{over:.1%} of the {n:,} documents are"
    return (f"Extracted text averages {t['tokens_per_word']:.2f} tokens per word with "
            f"{t['model'].split('/')[-1]}'s tokenizer, and {docs} longer than its "
            f"{t['window']}-token window ([CLS] and [SEP] included).\n\n"
            + _table(["Chunk / overlap (words)", "Chunks", f"Over {t['window']} tokens",
                      "Median tokens", "95th percentile"], body))


def zero_hit_table(r: Results) -> str:
    cols = [v for v in ("fts", "fts-or") if v in r.rows]
    buckets = {v: by_length(r.rows[v]) for v in cols}
    label = {"fts": "AND (default) finds nothing", "fts-or": "OR finds nothing"}
    body = []
    for length, _, _ in (*LENGTHS, ("all", 0, None)):
        def rows_of(v):
            return list(r.rows[v].values()) if length == "all" else buckets[v][length]
        n = len(rows_of(cols[0]))
        body.append([length, str(n), *(_share(sum(x["n_fts"] == 0 for x in rows_of(v)), n)
                                       for v in cols)])
    return _table(["Query length (words)", "Queries", *(label[v] for v in cols)], body)


def or_table(r: Results) -> str:
    """H6, plus the AND-then-OR fallback the decision rule scores."""
    hy, hor = r.rows["hybrid"], r.rows["hybrid-or"]
    fb = fallback_rows(hy, hor)
    long_ = [q for q, x in hy.items() if x["n_words"] >= 5]
    short = [q for q, x in hy.items() if x["n_words"] < 5]
    body = []
    for label, other in (("`hybrid-or`", hor), ("AND, then OR if AND finds nothing", fb)):
        for subset, qids in (("all", list(hy)), ("5+ words", long_), ("1–4 words", short)):
            c = compare(hy, other, qids)
            if c is None:
                continue
            body.append([label, subset, str(c[3]), _num(_mean(hy, qids)), _num(_mean(other, qids)),
                         _ci(*c[:3])])
    lat = np.asarray([x["latency_ms"] for x in fb.values()])
    switched = sum(x["n_fts"] == 0 for x in hy.values())
    return (_table(["Compared with `hybrid`", "Queries", "n", "`hybrid` nDCG@10", "nDCG@10",
                    "Δ [95% CI]"], body)
            + f"\n\nThe fallback runs OR for {_share(switched, len(hy))} queries. Its latency, "
              f"from the same rows (so without the failed AND attempt): p50 "
              f"{np.percentile(lat, 50):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms.")


def matched_in_table(r: Results) -> str:
    body = []
    groups = [("`hybrid`", r.rows["hybrid"]),
              ("`hybrid`, queries FTS matched",
               {q: x for q, x in r.rows["hybrid"].items() if x["n_fts"] > 0})]
    if "hybrid-or" in r.rows:
        groups.append(("`hybrid-or`", r.rows["hybrid-or"]))
    for label, rows in groups:
        p = matched_in_precision(rows, r.qrels)
        for b, _ in BUCKETS:
            cells = []
            for band in ("all", "1–3", "4–10"):
                n, rel = p[b][band]
                cells.append(f"{rel / n:.2f} ({rel}/{n})" if n else "—")
            body.append([label, {"both": "both", "fts": "FTS only", "vec": "vector only"}[b],
                         *cells])
    gaps = []
    for label, rows in groups:
        cells = []
        for _, lo, hi in RANK_BANDS:
            g = precision_gap(rows, r.qrels, lo, hi)
            cells.append(_ci(*g) if g else "—")
        gaps.append([label, *cells])
    return (_table(["Rows", "Best chunk found by", "Precision, top 10", "Ranks 1–3",
                    "Ranks 4–10"], body)
            + "\n\nPrecision of documents found by both minus those found by one, with a 95% "
              f"CI from resampling queries (— where either kind has fewer than {MIN_HITS} "
              "hits):\n\n"
            + _table(["Rows", "Ranks 1–3", "Ranks 4–10"], gaps)
            + "\n\nPrecision: the share of returned documents graded relevant (1 or 2), with "
              "the count; unjudged documents count as not relevant. Each document counts once, "
              "by the retrievers that found its best chunk among the 300 candidates each "
              "fetched.")


def by_type_table(r: Results) -> str:
    types = {t: [q for q, m in r.query_meta.items() if m["type"] == t] for t in TYPES}
    head = ["Variant", *(f"{t} (n = {len(_defined({q: r.rows['hybrid'][q] for q in qs}))})"
                         for t, qs in types.items())]
    means = _table(head, [[f"`{name}`", *(_num(_mean(r.rows[name], qs)) for qs in types.values())]
                          for name in r.variants])
    pairs = [("fts", "vec"), ("hybrid", "fts"), ("hybrid", "vec")]
    diffs = []
    for a, b in pairs:
        cells = []
        for qs in types.values():
            c = compare(r.rows[b], r.rows[a], qs)
            cells.append(_ci(*c[:3]) if c else "—")
        diffs.append([f"`{a}` − `{b}`", *cells])
    return (means + "\n\nPaired differences in nDCG@10 [95% CI]:\n\n"
            + _table(["Difference", *TYPES], diffs))


def by_author_table(r: Results) -> str:
    groups = {a: [q for q, m in r.query_meta.items() if m["type"] == "paraphrase" and m["author"] == a]
              for a in ("human", "llm-draft")}
    body = []
    for a, qs in groups.items():
        words = [r.rows["hybrid"][q]["n_words"] for q in qs]
        zero = sum(r.rows["hybrid"][q]["n_fts"] == 0 for q in qs)
        body.append([f"`{a}`", str(len(qs)), f"{min(words)}–{max(words)}",
                     _share(zero, len(qs)),
                     *(_num(_mean(r.rows[v], qs)) for v in ("fts", "vec", "hybrid"))])
    return _table(["Paraphrases by", "Queries", "Words", "FTS finds nothing",
                   "`fts`", "`vec`", "`hybrid`"], body)


def no_answer_table(r: Results) -> str:
    rows, k = r.rows["hybrid"], r.summary["variants"]["hybrid"]["search"]["rrf_k"]
    answerable = {q for q, judged in r.qrels.items() if any(g >= 1 for g in judged.values())}
    groups = {"answerable": [q for q in rows if q in answerable],
              "nothing relevant": [q for q in rows if q not in answerable]}
    body = []
    for label, qs in groups.items():
        scores = [s for q in qs if (s := top_hit_score(rows[q], k)) is not None]
        both = sum(rows[q]["hits"] and len(rows[q]["hits"][0]["matched_in"]) == 2 for q in qs)
        body.append([label, str(len(qs)), _share(sum(rows[q]["n_fts"] > 0 for q in qs), len(qs)),
                     _share(both, len(qs)),
                     " / ".join(_num(x) for x in (min(scores), statistics.median(scores), max(scores)))])
    worst = max(top_hit_score(rows[q], k) for q in groups["nothing relevant"])
    below = sum(top_hit_score(rows[q], k) <= worst for q in groups["answerable"])
    retyped = sorted(q for q in answerable if r.query_meta.get(q, {}).get("type") == "no-answer")
    note = (f"\n\n{_share(below, len(groups['answerable']))} answerable queries score no higher "
            "than the best-scoring query with nothing relevant, so no threshold on the score "
            f"separates the two without losing at least that many."
            if below else "\n\nEvery answerable query scores above every query with nothing relevant.")
    if retyped:
        note += (f" {', '.join(retyped)} {'was' if len(retyped) == 1 else 'were'} written as "
                 "`no-answer` but had a paper graded relevant, so counts as answerable.")
    return _table(["`hybrid`, queries with", "n", "FTS finds something", "Top hit found by both",
                   "Top hit's fused score: min / median / max"], body) + note


def rules_table(r: Results) -> str:
    v = r.summary["variants"]
    hy = r.rows["hybrid"]

    def ci(name):
        d = v[name]["vs_baseline"]["ndcg@10"]
        return d["diff"], d["lo"], d["hi"]

    fb = compare(hy, fallback_rows(hy, r.rows["hybrid-or"]))[:3]
    rerank, c200, c300 = ci("hybrid+rerank"), ci("hybrid-c200"), ci("hybrid-c300")
    met = {True: "rule met", False: "rule not met"}
    body = [
        ["Reranking in the MCP tools", "`hybrid+rerank`'s CI entirely above 0",
         _ci(*rerank), met[rerank[1] > 0]],
        ["AND, then OR, as the default", "the fallback's CI lower bound above −0.02",
         _ci(*fb), met[fb[1] > -0.02]],
        ["Chunk size", "`hybrid-c200`'s or `hybrid-c300`'s CI entirely above 0",
         f"c200 {_ci(*c200)}; c300 {_ci(*c300)}", met[c200[1] > 0 or c300[1] > 0]],
    ]
    return _table(["Decision", "Rule, set before the grades", "Δ nDCG@10 vs `hybrid` [95% CI]",
                   "Result"], body)


# name -> (render, built on grades?)
BLOCKS: dict[str, tuple[Callable[[Results], str], bool]] = {
    "main": (main_table, True),
    "cost": (cost_table, False),
    "tokens": (tokens_table, False),
    "zero-hit": (zero_hit_table, False),
    "or": (or_table, True),
    "matched-in": (matched_in_table, True),
    "by-type": (by_type_table, True),
    "by-author": (by_author_table, True),
    "no-answer": (no_answer_table, True),
    "rules": (rules_table, True),
}

_BLOCK = re.compile(r"(<!-- report:(\S+) (\S+) -->\n)(.*?)(<!-- /report -->)", re.S)


def render(text: str, results: Mapping[str, Results], provisional: bool = False,
           ) -> tuple[str, list[str]]:
    """``text`` with every block regenerated, and the blocks left alone
    because their dataset isn't fully judged yet."""
    opened = len(re.findall(r"<!-- report:\S+ \S+ -->", text))
    if opened != len(_BLOCK.findall(text)):
        raise ValueError("unclosed <!-- report:… --> block: each needs a <!-- /report --> line")
    skipped: list[str] = []

    def fill(m: re.Match) -> str:
        start, name, dataset, old, end = m.groups()
        if name not in BLOCKS:
            raise ValueError(f"unknown report block {name!r}; known: {', '.join(BLOCKS)}")
        if dataset not in results:
            raise ValueError(f"block {name} is for {dataset!r}, which has no results loaded")
        fn, graded = BLOCKS[name]
        r = results[dataset]
        if graded and not r.complete:
            if not provisional:
                skipped.append(f"{name} {dataset}")
                return m.group(0)
            return (f"{start}> **Provisional:** not every variant's top 10 is judged yet.\n\n"
                    f"{fn(r)}\n{end}")
        return f"{start}{fn(r)}\n{end}"

    return _BLOCK.sub(fill, text), skipped


LABELS = {"scifact": "SciFact", "papers": "Papers set"}


def write_charts(r: Results, out_dir: Path) -> list[Path]:
    from evals import charts

    s, base = r.summary, r.summary["baseline"]
    label = ("Provisional: " if not r.complete else "") + LABELS.get(r.name, r.name)
    diffs = [(name, d["diff"], d["lo"], d["hi"]) for name in r.variants if name != base
             for d in [s["variants"][name]["vs_baseline"]["ndcg@10"]]]
    points = [(name, v["latency_ms"]["p50"], v["latency_ms"]["p95"], v["metrics"]["ndcg@10"]["mean"])
              for name in r.variants for v in [s["variants"][name]]]
    n = s["variants"][base]["metrics"]["ndcg@10"]["n"]
    cpus = ", ".join(sorted({s["variants"][v]["environment"]["cpu"] for v in r.variants}))
    written = []
    for theme_name, theme in charts.THEMES.items():
        suffix = "" if theme_name == "light" else "-dark"
        path = out_dir / f"ndcg-{r.name}{suffix}.svg"
        charts.difference_chart(
            diffs, baseline=base, title=f"{label}: nDCG@10 difference from {base}",
            subtitle=f"Mean over {n} queries, with its paired bootstrap 95% CI",
            theme=theme, path=path)
        written.append(path)
        path = out_dir / f"latency-{r.name}{suffix}.svg"
        charts.latency_chart(
            points, baseline=base, title=f"{label}: nDCG@10 against latency per query",
            subtitle=f"Dot at p50, line to p95; warm, on {cpus.removeprefix('AMD ')}",
            theme=theme, path=path)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m evals.report", description=__doc__.split("\n")[0])
    p.add_argument("--report", type=Path, default=REPO / "evals" / "REPORT.md")
    p.add_argument("--results", type=Path, default=REPO / "evals" / "results")
    p.add_argument("--charts", type=Path, default=REPO / "evals" / "results" / "charts")
    p.add_argument("--cache", type=Path, default=REPO / "evals" / ".cache")
    p.add_argument("--provisional", action="store_true",
                   help="render blocks and charts built on grades before every variant is judged")
    args = p.parse_args(argv)

    text = args.report.read_text()
    wanted = sorted(set(m.group(3) for m in _BLOCK.finditer(text)))
    results = {name: load(name, args.results, args.cache) for name in wanted}
    out, skipped = render(text, results, args.provisional)
    args.report.write_text(out)
    print(f"{args.report}: {len(_BLOCK.findall(out)) - len(skipped)} block(s) written")
    for r in results.values():
        if r.complete or args.provisional:
            print(f"charts: {', '.join(p.name for p in write_charts(r, args.charts))}")
        else:
            skipped.append(f"the {r.name} charts")
    if skipped:
        print(f"left as they were, until every variant is judged: {', '.join(skipped)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
