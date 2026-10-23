"""Run retrieval variants over a dataset; write per-query rows and a summary.

    uv run python -m evals.run --dataset scifact --variants all

Each query goes through ``alexandria.search.search(limit=100)``; the chunk
hits are collapsed to documents in order of first appearance and the top 10
documents are scored. Rows go to ``<out>/<dataset>/<variant>.jsonl`` and the
aggregate to ``<out>/<dataset>/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import subprocess
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import numpy as np

from alexandria.search import search
from evals.datasets import Dataset, scifact
from evals.index import BuiltIndex, build_index, canonical_ids, open_index, translate_qrels
from evals.metrics import first_unique, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k
from evals.variants import BASELINE, Variant, resolve

SEARCH_LIMIT = 100   # chunks fetched per query
DOC_K = 10           # documents scored per query
METRICS = ("ndcg@10", "recall@5", "recall@10", "mrr@10")
BOOTSTRAP = {"n_resamples": 10_000, "seed": 0, "alpha": 0.05}

REPO = Path(__file__).resolve().parent.parent
DATASETS: dict[str, Callable[[Path], Dataset]] = {"scifact": scifact.load}


def _score(docs: list[str], judged: dict[str, int]) -> dict[str, float | None]:
    return {
        "ndcg@10": ndcg_at_k(docs, judged, k=10),
        "recall@5": recall_at_k(docs, judged, k=5),
        "recall@10": recall_at_k(docs, judged, k=10),
        "mrr@10": mrr_at_k(docs, judged, k=10),
    }


def run_variant(ds: Dataset, v: Variant, built: BuiltIndex) -> list[dict]:
    """One row per query, in dataset order. Latency is warm: one untimed
    query first loads the embedding model."""
    canon = canonical_ids(built.doc_map)
    qrels = translate_qrels(ds.qrels, built.doc_map)
    cfg, conn = open_index(built, query_instruction=v.query_instruction)
    rows = []
    try:
        if ds.queries:
            search(next(iter(ds.queries.values())), conn, cfg, limit=SEARCH_LIMIT, mode=v.mode)
        for qid, text in ds.queries.items():
            start = time.perf_counter()
            hits = search(text, conn, cfg, limit=SEARCH_LIMIT, mode=v.mode)
            latency_ms = (time.perf_counter() - start) * 1000
            docs = first_unique((canon[h.doc_id] for h in hits), DOC_K)
            first_hit = {}
            for h in hits:
                first_hit.setdefault(canon[h.doc_id], h)
            # A chunk's position in its document, unlike its id, is the same
            # in every build of the index.
            ords = {
                d: conn.execute(
                    "SELECT ord FROM chunks WHERE id = ?", (first_hit[d].chunk_id,)
                ).fetchone()[0]
                for d in docs
            }
            rows.append({
                "query_id": qid,
                "variant": v.name,
                "mode": v.mode,
                "n_words": len(text.split()),
                "latency_ms": round(latency_ms, 2),
                "docs": docs,
                "hits": [
                    {
                        "doc": d,
                        "chunk": ords[d],
                        "fts_rank": first_hit[d].fts_rank,
                        "vec_rank": first_hit[d].vec_rank,
                        "matched_in": list(first_hit[d].matched_in),
                    }
                    for d in docs
                ],
                "n_hits": len(hits),
                "n_fts": sum(h.fts_rank is not None for h in hits),
                "n_vec": sum(h.vec_rank is not None for h in hits),
                **_score(docs, qrels.get(qid, {})),
            })
    finally:
        conn.close()
    return rows


def _git_sha(repo: Path = REPO) -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        # Results are rewritten by every run (latency), so they don't count;
        # new, uncommitted files do.
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)evals/results"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("+dirty" if dirty else "")


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment() -> dict:
    """What produced the numbers. No hostname or user paths."""
    return {
        "cpu": _cpu(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        **{
            pkg: version(pkg)
            for pkg in ("sqlite-vec", "torch", "sentence-transformers", "transformers",
                        "tokenizers", "numpy")
        },
        "alexandria_commit": _git_sha(),
    }


def summarize(
    ds: Dataset,
    rows_by_variant: dict[str, list[dict]],
    builds: dict[str, dict],
    envs: dict[str, dict] | None = None,
) -> dict:
    """``envs`` maps each variant to the environment that produced its rows
    (variants can come from different runs); missing ones get the current one."""
    envs = envs or {}
    variants = {}
    base_rows = {r["query_id"]: r for r in rows_by_variant.get(BASELINE, [])}
    for name, rows in rows_by_variant.items():
        metrics = {}
        for m in METRICS:
            vals = [r[m] for r in rows if r[m] is not None]
            metrics[m] = {"mean": float(np.mean(vals)) if vals else None, "n": len(vals)}
        lat = [r["latency_ms"] for r in rows]
        uses_fts = bool(rows) and rows[0]["mode"] != "vec"
        entry = {
            "metrics": metrics,
            "latency_ms": {
                "p50": float(np.percentile(lat, 50)) if lat else None,
                "p95": float(np.percentile(lat, 95)) if lat else None,
            },
            "fts_zero_hit_rate": (
                sum(r["n_fts"] == 0 for r in rows) / len(rows) if rows and uses_fts else None
            ),
            "index": builds.get(name, {}),
            "environment": envs[name] if name in envs else environment(),
        }
        if base_rows and name != BASELINE:
            entry["vs_baseline"] = {}
            for m in METRICS:
                pairs = [
                    (base_rows[r["query_id"]][m], r[m]) for r in rows
                    if r["query_id"] in base_rows
                    and r[m] is not None and base_rows[r["query_id"]][m] is not None
                ]
                if not pairs:
                    continue
                diff, lo, hi = paired_bootstrap(
                    [b for b, _ in pairs], [o for _, o in pairs], **BOOTSTRAP
                )
                entry["vs_baseline"][m] = {"diff": diff, "lo": lo, "hi": hi, "n": len(pairs)}
        variants[name] = entry
    return {
        "dataset": ds.name,
        "source": ds.source,
        "n_queries": len(ds.queries),
        "baseline": BASELINE,
        "protocol": {"search_limit": SEARCH_LIMIT, "doc_k": DOC_K, "bootstrap": BOOTSTRAP},
        "variants": variants,
    }


def _cell(x: float | None, spec: str, width: int) -> str:
    return f"{'—' if x is None else format(x, spec):>{width}}"


def format_table(summary: dict) -> str:
    head = f"{'variant':<14}{'nDCG@10':>9}{'R@5':>8}{'R@10':>8}{'MRR@10':>8}" \
           f"{'Δ nDCG@10 [95% CI]':>28}{'p50 ms':>9}{'p95 ms':>9}{'FTS 0-hit':>11}"
    lines = [head, "-" * len(head)]
    for name, v in summary["variants"].items():
        m = {k: v["metrics"][k]["mean"] for k in METRICS}
        d = v.get("vs_baseline", {}).get("ndcg@10")
        if d:
            delta = f"{d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
        else:
            delta = "baseline" if name == summary["baseline"] else "—"
        zero = v["fts_zero_hit_rate"]
        lines.append(
            f"{name:<14}{_cell(m['ndcg@10'], '.4f', 9)}{_cell(m['recall@5'], '.4f', 8)}"
            f"{_cell(m['recall@10'], '.4f', 8)}{_cell(m['mrr@10'], '.4f', 8)}{delta:>28}"
            f"{_cell(v['latency_ms']['p50'], '.1f', 9)}{_cell(v['latency_ms']['p95'], '.1f', 9)}"
            f"{'' if zero is None else f'{zero:.1%}':>11}"
        )
    return "\n".join(lines)


def _previous_results(out_dir: Path) -> tuple[dict, dict, dict]:
    """Rows, build stats and environments of variants already in ``out_dir``,
    so running a subset of variants updates the summary instead of replacing it."""
    old = out_dir / "summary.json"
    old_variants = json.loads(old.read_text())["variants"] if old.exists() else {}
    rows, builds, envs = {}, {}, {}
    for path in sorted(out_dir.glob("*.jsonl")):
        rows[path.stem] = [json.loads(line) for line in path.read_text().splitlines()]
        entry = old_variants.get(path.stem, {})
        builds[path.stem] = entry.get("index", {})
        envs[path.stem] = entry.get("environment")  # None: produced by an unknown run
    return rows, builds, envs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--variants", default="all", help='"all" or a comma-separated list')
    p.add_argument("--out", type=Path, default=REPO / "evals" / "results")
    p.add_argument("--cache", type=Path, default=REPO / "evals" / ".cache")
    args = p.parse_args(argv)

    env = environment()  # before the run: a commit made during it isn't what produced it
    ds = DATASETS[args.dataset](args.cache)
    out_dir = args.out / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_variant, builds, envs = _previous_results(out_dir)
    for v in resolve(args.variants):
        built = build_index(ds, v.index, args.cache)
        rows = run_variant(ds, v, built)
        rows_by_variant[v.name], builds[v.name], envs[v.name] = rows, built.stats, env
        with (out_dir / f"{v.name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"{v.name}: {len(rows)} queries", flush=True)

    summary = summarize(ds, rows_by_variant, builds, envs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(format_table(summary))


if __name__ == "__main__":
    main()
