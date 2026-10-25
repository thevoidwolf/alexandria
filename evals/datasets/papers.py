"""A public set of arXiv papers on agents, evals, retrieval and instruction-following.

    uv run python -m evals.datasets.papers                   # fetch, verify, pin
    uv run python -m evals.datasets.papers --update 2104.08663   # re-pin one paper
    uv run python -m evals.datasets.papers --update          # re-pin every paper

Committed, in ``evals/data/papers/``:

- ``papers.tsv``: ``arxiv_id <TAB> label``, a short human name for each paper.
- ``papers.lock.json``: for each paper the arXiv version, exact title, authors
  and the PDF's sha256, so every clone measures the same corpus.
- ``queries.jsonl`` and ``qrels.tsv``: Voidwolf's queries and judgements.

The PDFs themselves go to ``<cache>/papers/`` and are never committed: arXiv's
default licence doesn't cover redistribution.

Fetching fails loudly. Every problem is collected and reported together, and
the lock is left untouched, when an id doesn't resolve on arXiv, a label
doesn't match its paper's title, or a pinned paper's title or PDF bytes no
longer match the lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from evals.datasets import Dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "papers"
LOCK_NAME = "papers.lock.json"
API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://export.arxiv.org/pdf/{id}v{version}"
REQUEST_GAP_SECONDS = 3.0           # arXiv asks for at most one request every 3 s
QUERY_TYPES = ("lookup", "paraphrase", "conceptual", "no-answer")
QUERY_AUTHORS = ("human", "llm-draft")

_ID = re.compile(r"\d{4}\.\d{4,5}")
_QUERY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_ABS_ID = re.compile(r"arxiv\.org/abs/(?P<id>\d{4}\.\d{4,5})v(?P<v>\d+)$")
_ATOM = "{http://www.w3.org/2005/Atom}"


class FetchError(Exception):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Entry:
    arxiv_id: str
    version: int
    title: str
    authors: tuple[str, ...]


# ------------------------------------------------------------------ labels

_STOPWORDS = frozenset({"a", "an", "and", "the", "for", "of", "on", "in", "to", "with"})
_EXPANSIONS = {
    "llm": {"large", "language", "model"},
    "llms": {"large", "language", "models"},
    "rag": {"retrieval", "augmented", "generation"},
}
_NOTE = re.compile(r"^(?P<main>.*?)\s*\((?P<note>[^()]*)\)$")
_ET_AL = re.compile(r"^(?P<name>.+?)\s+et\s+al\.?$", re.IGNORECASE)


def _words(s: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", s).casefold())


def _fits(part: str, title_words: set[str]) -> bool:
    """Every content word of ``part`` is in the title (or spelled out there)."""
    words = [w for w in _words(part) if w not in _STOPWORDS]
    return bool(words) and all(
        w in title_words or (w in _EXPANSIONS and _EXPANSIONS[w] <= title_words)
        for w in words
    )


def label_matches(label: str, title: str, authors: Sequence[str]) -> bool:
    """Whether a papers.tsv label plausibly names the paper with this title.

    A label is a short name with an optional note in parentheses. It matches
    when every content word of the name, or of the note, appears in the
    title; LLM(s) and RAG may stand for their spelled-out forms. A note of
    the form "Surname et al." instead requires the name to match *and* that
    surname among the authors. Tested against the committed set: each label
    matches its own paper and none of the others.
    """
    title_words = set(_words(title))
    m = _NOTE.match(label.strip())
    main, note = (m["main"], m["note"].strip()) if m else (label, None)
    if note is not None and (cited := _ET_AL.match(note)):
        surname = _words(cited["name"])
        by_them = any(surname and _words(a)[-len(surname):] == surname for a in authors)
        return _fits(main, title_words) and by_them
    return _fits(main, title_words) or (note is not None and _fits(note, title_words))


# ------------------------------------------------------------ files on disk

def read_papers(path: Path) -> dict[str, str]:
    """papers.tsv -> {arxiv_id: label}, in file order."""
    papers: dict[str, str] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not _ID.fullmatch(parts[0].strip()) or not parts[1].strip():
            raise ValueError(f"{path.name} line {n}: expected '<arxiv id><TAB><label>', got {line!r}")
        arxiv_id, label = parts[0].strip(), parts[1].strip()
        if arxiv_id in papers:
            raise ValueError(f"{path.name} line {n}: {arxiv_id} is listed twice")
        papers[arxiv_id] = label
    return papers


def _read_lock(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_queries(path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    queries: dict[str, str] = {}
    meta: dict[str, dict] = {}
    if not path.exists():
        return queries, meta
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name} line {n}"
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}: not valid JSON ({e})") from None
        if not isinstance(q, dict):
            raise ValueError(f"{where}: expected a JSON object")
        qid, text = q.get("id"), q.get("text")
        if not isinstance(qid, str) or not _QUERY_ID.fullmatch(qid):
            # Ids end up in qrels.tsv: no spaces, tabs or slashes.
            raise ValueError(
                f"{where}: id must be letters, digits, '.', '_' or '-', starting with a "
                f"letter or digit; got {qid!r}"
            )
        if qid in queries:
            raise ValueError(f"{where}: id {qid!r} is used twice")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}: text is empty")
        if q.get("type") not in QUERY_TYPES:
            raise ValueError(f"{where}: type must be one of {', '.join(QUERY_TYPES)}; got {q.get('type')!r}")
        if q.get("author") not in QUERY_AUTHORS:
            raise ValueError(f"{where}: author must be one of {', '.join(QUERY_AUTHORS)}; got {q.get('author')!r}")
        queries[qid] = text
        meta[qid] = {"type": q["type"], "author": q["author"]}
    return queries, meta


def _read_qrels(path: Path, queries: dict[str, str], papers: dict[str, str]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    if not path.exists():
        return qrels
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name} line {n}"
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{where}: expected '<query id><TAB><arxiv id><TAB><grade>', got {line!r}")
        qid, doc, grade = (p.strip() for p in parts)
        if qid not in queries:
            raise ValueError(f"{where}: unknown query {qid!r}")
        if doc not in papers:
            raise ValueError(f"{where}: {doc} is not in papers.tsv")
        if grade not in ("0", "1", "2"):
            raise ValueError(f"{where}: grade must be 0, 1 or 2; got {grade!r}")
        if doc in qrels.get(qid, {}):
            raise ValueError(f"{where}: {qid} / {doc} is judged twice")
        qrels.setdefault(qid, {})[doc] = int(grade)
    return qrels


# ----------------------------------------------------------------- arXiv

def parse_atom(xml: bytes) -> dict[str, Entry]:
    """arXiv API response -> {arxiv_id: Entry}. Error entries are skipped,
    and an id arXiv doesn't know is simply absent from the response."""
    out: dict[str, Entry] = {}
    for e in ET.fromstring(xml).iter(f"{_ATOM}entry"):
        m = _ABS_ID.search((e.findtext(f"{_ATOM}id") or "").strip())
        if not m:
            continue
        out[m["id"]] = Entry(
            arxiv_id=m["id"],
            version=int(m["v"]),
            title=" ".join((e.findtext(f"{_ATOM}title") or "").split()),
            authors=tuple(
                " ".join((a.text or "").split()) for a in e.findall(f"{_ATOM}author/{_ATOM}name")
            ),
        )
    return out


def _http_get() -> Callable[[str], bytes]:
    from alexandria.config import HttpConfig

    client = httpx.Client(
        headers={"User-Agent": f"{HttpConfig().user_agent} retrieval-evals"},
        timeout=60, follow_redirects=True,
    )

    def get(url: str) -> bytes:
        r = client.get(url)
        r.raise_for_status()
        return r.content

    return get


def fetch(
    data_dir: Path,
    cache_dir: Path,
    *,
    http_get: Callable[[str], bytes] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    update: bool | Iterable[str] = False,
    write_lock: bool = True,
) -> dict[str, dict]:
    """Verify every paper against arXiv, download missing PDFs, write the lock.

    Pinned papers are checked at their locked version. ``update`` re-resolves
    papers to their latest version instead: ``True`` for all of them, or an
    iterable of ids for just those. Raises ``FetchError`` listing every
    problem, without touching the lock, if anything doesn't check out.
    ``write_lock=False`` checks and downloads but never writes the lock.
    """
    labels = read_papers(data_dir / "papers.tsv")
    lock_path = data_dir / LOCK_NAME
    old = _read_lock(lock_path)
    repin = set(labels) if update is True else set(update or ())
    if unknown := sorted(repin - set(labels)):
        raise ValueError(f"can't update papers that aren't in papers.tsv: {', '.join(unknown)}")
    get = http_get or _http_get()
    pinned = {i: old[i]["version"] for i in labels if i in old and i not in repin}

    ids = [f"{i}v{pinned[i]}" if i in pinned else i for i in labels]
    entries = parse_atom(get(f"{API_URL}?id_list={','.join(ids)}&max_results={len(ids)}"))

    problems: list[str] = []
    for i, label in labels.items():
        e = entries.get(i)
        if e is None:
            problems.append(f"{i}: arXiv has no paper with this id (papers.tsv label {label!r})")
            continue
        if not label_matches(label, e.title, e.authors):
            problems.append(f"{i}: papers.tsv label {label!r} doesn't match the arXiv title {e.title!r}")
        if i in pinned and e.title != old[i]["title"]:
            problems.append(
                f"{i}v{e.version}: arXiv now titles it {e.title!r}; the lock says {old[i]['title']!r}"
            )
    if problems:
        raise FetchError(problems)

    pdf_dir = cache_dir / "papers"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    lock: dict[str, dict] = {}
    for i in labels:
        e = entries[i]
        path = pdf_dir / f"{i}v{e.version}.pdf"
        cached = path.exists()
        if cached:
            data = path.read_bytes()
        else:
            sleep(REQUEST_GAP_SECONDS)
            try:
                data = get(PDF_URL.format(id=i, version=e.version))
            except httpx.HTTPError as err:
                problems.append(f"{i}v{e.version}: PDF download failed ({err})")
                continue
        where = f"the cached copy {path}" if cached else "arXiv's copy"
        fix = "; delete it to re-download" if cached else ""
        if not data.startswith(b"%PDF"):
            problems.append(f"{i}v{e.version}: {where} is not a PDF{fix}")
            continue
        if b"%%EOF" not in data[-2048:]:
            problems.append(f"{i}v{e.version}: {where} looks truncated (no %%EOF at the end){fix}")
            continue
        sha = hashlib.sha256(data).hexdigest()
        if i in pinned and sha != old[i]["sha256"]:
            problems.append(
                f"{i}v{e.version}: sha256 of {where} is {sha[:12]}…, the lock says "
                f"{old[i]['sha256'][:12]}…; delete a bad cached copy, or check the paper and "
                f"rerun with `--update {i}` to re-pin just this paper"
            )
            continue
        if not cached:
            # Write beside and rename, so an interrupted write never leaves a
            # partial PDF at the path a later run would trust.
            part = path.with_name(path.name + ".part")
            part.write_bytes(data)
            os.replace(part, path)
        lock[i] = {"version": e.version, "title": e.title, "authors": list(e.authors), "sha256": sha}
    if problems:
        raise FetchError(problems)

    if write_lock and lock != old:
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    return lock


# ------------------------------------------------------------------ loading

def load(cache_dir: Path, data_dir: Path = DATA_DIR) -> Dataset:
    labels = read_papers(data_dir / "papers.tsv")
    lock_path = data_dir / LOCK_NAME
    lock = _read_lock(lock_path)
    unpinned = [i for i in labels if i not in lock]
    if unpinned:
        raise ValueError(
            f"not in {LOCK_NAME}: {', '.join(unpinned)}; run "
            "`uv run python -m evals.datasets.papers` to fetch, check and pin them"
        )

    def pdf(i: str) -> Path:
        return cache_dir / "papers" / f"{i}v{lock[i]['version']}.pdf"

    if any(not pdf(i).exists() for i in labels):
        # Only fills the cache: an eval run never rewrites the committed lock.
        fetch(data_dir, cache_dir, http_get=_http_get(), sleep=time.sleep, write_lock=False)

    docs: dict[str, str | bytes] = {}
    for i in labels:
        data = pdf(i).read_bytes()
        if hashlib.sha256(data).hexdigest() != lock[i]["sha256"]:
            raise ValueError(f"{pdf(i)} doesn't match its sha256 in {LOCK_NAME}; delete it to re-download")
        docs[i] = data

    queries, meta = _read_queries(data_dir / "queries.jsonl")
    return Dataset(
        name="papers",
        docs=docs,
        queries=queries,
        qrels=_read_qrels(data_dir / "qrels.tsv", queries, labels),
        source={LOCK_NAME: hashlib.sha256(lock_path.read_bytes()).hexdigest()},
        doc_suffix=".pdf",
        titles={i: lock[i]["title"] for i in labels},
        query_meta=meta,
        pooled=True,
    )


def title_lookups(papers: Dataset) -> Dataset:
    """Known-item searches by title (H7): one query per paper, its arXiv title
    typed as is, with that paper as its only relevant document."""
    ids = sorted(papers.titles)
    queries = {f"t{n:02d}": papers.titles[i] for n, i in enumerate(ids, 1)}
    return Dataset(
        name="papers-titles",
        docs=papers.docs,
        queries=queries,
        qrels={q: {i: 2} for q, i in zip(queries, ids)},
        source=papers.source,
        doc_suffix=papers.doc_suffix,
        titles=papers.titles,
    )


def load_titles(cache_dir: Path, data_dir: Path = DATA_DIR) -> Dataset:
    return title_lookups(load(cache_dir, data_dir))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m evals.datasets.papers",
        description="Fetch the arXiv papers set, check it against papers.tsv and pin it.",
    )
    p.add_argument("--update", nargs="*", metavar="ID",
                   help="re-pin these papers (or, with no ids, every paper) to their latest "
                        "arXiv version")
    p.add_argument("--cache", type=Path, default=DATA_DIR.parents[1] / ".cache")
    args = p.parse_args(argv)
    try:
        update = True if args.update == [] else (args.update or False)
        lock = fetch(DATA_DIR, args.cache, update=update)
    except (FetchError, ValueError) as e:
        problems = e.problems if isinstance(e, FetchError) else [str(e)]
        print(f"papers set NOT fetched: {len(problems)} problem(s)", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        sys.exit(1)
    for i, entry in lock.items():
        print(f"{i}v{entry['version']:<3} {entry['title']}")
    print(f"{len(lock)} papers checked and pinned in {DATA_DIR / LOCK_NAME}")


if __name__ == "__main__":
    main()
