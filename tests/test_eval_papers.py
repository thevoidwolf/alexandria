"""The arXiv papers set: label checks, pinned fetching that fails loudly, loading."""
import hashlib
import json
import re
from urllib.parse import parse_qs, urlparse
from xml.sax.saxutils import escape

import httpx
import pytest

from evals.datasets import papers
from evals.datasets.papers import FetchError, fetch, label_matches, load, parse_atom

# ---------------------------------------------------------------- label rule


@pytest.mark.parametrize("label,title,authors", [
    ("SysBench", "SysBench: Can Large Language Models Follow System Messages?", []),
    ("RuLES: Can LLMs Follow Simple Rules?", "Can LLMs Follow Simple Rules?", []),
    # nickname first, the title's own words in the note
    ("ALCE (citations)", "Enabling Large Language Models to Generate Text with Citations", []),
    # abbreviations the title spells out
    ("RAG for LLMs: A Survey", "Retrieval-Augmented Generation for Large Language Models: A Survey", []),
    ("The Refusal–Compliance Tradeoff",
     "The Refusal--Compliance Tradeoff: A Large-Scale Safety Behavior Audit", []),
    ("Retrieval-Augmented Generation (Lewis et al.)",
     "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
     ["Patrick Lewis", "Ethan Perez"]),
])
def test_label_matches_its_paper(label, title, authors):
    assert label_matches(label, title, authors)


@pytest.mark.parametrize("label,title,authors", [
    ("BIPIA", "Benchmarking and Defending Against Indirect Prompt Injection Attacks", []),
    ("MTEB", "C-Pack: Packed Resources For General Chinese Embeddings", []),
    ("The Instruction Hierarchy", "Control Illusion: The Failure of Instruction Hierarchies", []),
    # the main part fits, but the cited author didn't write this one
    ("Retrieval-Augmented Generation (Lewis et al.)",
     "Ragas: Automated Evaluation of Retrieval Augmented Generation", ["Shahul Es", "Jithin James"]),
    ("", "Anything", []),
])
def test_label_does_not_match_another_paper(label, title, authors):
    assert not label_matches(label, title, authors)


# ------------------------------------------------------------ Atom parsing

def _entry(arxiv_id, version, title, authors):
    names = "".join(f"<author><name>{escape(a)}</name></author>" for a in authors)
    return (f"<entry><id>http://arxiv.org/abs/{arxiv_id}v{version}</id>"
            f"<title>{escape(title)}</title>{names}</entry>")


def _feed(*entries):
    return ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
            + "".join(entries) + "</feed>").encode()


def test_parse_atom_reads_entries_and_skips_errors():
    xml = _feed(
        _entry("2311.04235", 3, "Can LLMs Follow\n   Simple Rules?", ["Norman Mu", "Sarah Chen"]),
        "<entry><id>https://arxiv.org/api/errors#incorrect_id_format_for_x</id>"
        "<title>Error</title></entry>",
    )
    got = parse_atom(xml)
    assert list(got) == ["2311.04235"]
    e = got["2311.04235"]
    assert (e.version, e.title, e.authors) == (3, "Can LLMs Follow Simple Rules?", ("Norman Mu", "Sarah Chen"))


# ---------------------------------------------------------------- fetching

def _pdf(tag: str) -> bytes:
    return b"%PDF-1.5\n" + tag.encode() + b"\n%%EOF\n"


class FakeArxiv:
    """Serves the API and versioned PDFs from a table: id -> [v1, v2, ...]."""

    def __init__(self, table):
        self.table = table
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url.startswith(papers.API_URL):
            entries = []
            for requested in parse_qs(urlparse(url).query)["id_list"][0].split(","):
                bare, _, v = requested.partition("v")
                versions = self.table.get(bare, [])
                n = int(v) if v else len(versions)
                if 1 <= n <= len(versions):
                    entries.append(_entry(bare, n, versions[n - 1]["title"], versions[n - 1]["authors"]))
            return _feed(*entries)
        m = re.fullmatch(r".*/pdf/(\d{4}\.\d{4,5})v(\d+)", url)
        if not m or m[1] not in self.table or int(m[2]) > len(self.table[m[1]]):
            raise httpx.HTTPError(f"404 for {url}")
        return self.table[m[1]][int(m[2]) - 1]["pdf"]

    def pdf_requests(self):
        return [c for c in self.calls if "/pdf/" in c]


TABLE = {
    "2311.04235": [
        {"title": "Can LLMs Follow Simple Rules?", "authors": ["Norman Mu"], "pdf": _pdf("rules v1")},
        {"title": "Can LLMs Follow Simple Rules?", "authors": ["Norman Mu"], "pdf": _pdf("rules v2")},
    ],
    "2104.08663": [
        {"title": "BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
         "authors": ["Nandan Thakur"], "pdf": _pdf("beir v1")},
    ],
}


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "papers.tsv").write_text(
        "2311.04235\tRuLES: Can LLMs Follow Simple Rules?\n2104.08663\tBEIR\n"
    )
    return d


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "cache"


def _no_sleep(seconds):
    pass


def test_fetch_pins_latest_versions_and_saves_pdfs(data_dir, cache):
    arxiv = FakeArxiv(TABLE)
    waits = []
    lock = fetch(data_dir, cache, http_get=arxiv, sleep=waits.append)

    assert lock["2311.04235"] == {
        "version": 2, "title": "Can LLMs Follow Simple Rules?", "authors": ["Norman Mu"],
        "sha256": hashlib.sha256(_pdf("rules v2")).hexdigest(),
    }
    assert json.loads((data_dir / "papers.lock.json").read_text()) == lock
    assert (cache / "papers" / "2311.04235v2.pdf").read_bytes() == _pdf("rules v2")
    assert len(waits) == 2 and set(waits) == {papers.REQUEST_GAP_SECONDS}   # one per PDF


def test_unresolved_id_fails_loudly_and_writes_nothing(data_dir, cache):
    (data_dir / "papers.tsv").write_text("2311.04235\tRuLES\n2609.99999\tA paper that isn't there\n")
    with pytest.raises(FetchError) as err:
        fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep)
    assert any("2609.99999" in p for p in err.value.problems)
    assert not (data_dir / "papers.lock.json").exists()


def test_label_that_does_not_match_the_title_fails_loudly(data_dir, cache):
    (data_dir / "papers.tsv").write_text("2104.08663\tMTEB\n")      # wrong id for this label
    with pytest.raises(FetchError) as err:
        fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep)
    assert "MTEB" in err.value.problems[0] and "BEIR" in err.value.problems[0]
    assert not (data_dir / "papers.lock.json").exists()


def test_malformed_id_is_caught_before_asking_arxiv(data_dir, cache):
    # arXiv answers a batch containing one malformed id with nothing but an error.
    (data_dir / "papers.tsv").write_text("2104.08663\tBEIR\n12345\tbad\n")
    arxiv = FakeArxiv(TABLE)
    with pytest.raises(ValueError, match="line 2"):
        fetch(data_dir, cache, http_get=arxiv, sleep=_no_sleep)
    assert arxiv.calls == []


def test_refetch_uses_the_pinned_version_and_the_cache(data_dir, cache):
    fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep)
    newer = {**TABLE, "2311.04235": TABLE["2311.04235"] + [
        {"title": "Can LLMs Follow Simple Rules?", "authors": ["Norman Mu"], "pdf": _pdf("rules v3")}]}
    arxiv = FakeArxiv(newer)
    lock = fetch(data_dir, cache, http_get=arxiv, sleep=_no_sleep)
    assert lock["2311.04235"]["version"] == 2
    assert "2311.04235v2" in arxiv.calls[0]
    assert arxiv.pdf_requests() == []


def test_pdf_that_differs_from_the_lock_fails_and_is_not_kept(data_dir, cache):
    fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep)
    before = (data_dir / "papers.lock.json").read_text()
    (cache / "papers" / "2104.08663v1.pdf").unlink()
    regenerated = {**TABLE, "2104.08663": [{**TABLE["2104.08663"][0], "pdf": _pdf("re-rendered")}]}
    with pytest.raises(FetchError) as err:
        fetch(data_dir, cache, http_get=FakeArxiv(regenerated), sleep=_no_sleep)
    assert any("2104.08663" in p and "sha256" in p for p in err.value.problems)
    assert (data_dir / "papers.lock.json").read_text() == before
    assert not (cache / "papers" / "2104.08663v1.pdf").exists()


def test_title_that_changed_for_a_pinned_version_fails(data_dir, cache):
    fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep)
    retitled = {**TABLE, "2104.08663": [{**TABLE["2104.08663"][0], "title": "BEIR: Renamed"}]}
    with pytest.raises(FetchError) as err:
        fetch(data_dir, cache, http_get=FakeArxiv(retitled), sleep=_no_sleep)
    assert any("BEIR: Renamed" in p for p in err.value.problems)


def test_update_moves_the_pin_to_the_latest_version(data_dir, cache):
    fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep)
    newer = {**TABLE, "2311.04235": TABLE["2311.04235"] + [
        {"title": "Can LLMs Follow Simple Rules?", "authors": ["Norman Mu"], "pdf": _pdf("rules v3")}]}
    lock = fetch(data_dir, cache, http_get=FakeArxiv(newer), sleep=_no_sleep, update=True)
    assert lock["2311.04235"]["version"] == 3


# ----------------------------------------------------------------- loading

@pytest.fixture
def fetched(data_dir, cache):
    fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep)
    return data_dir


def _write_queries(data_dir, *rows):
    (data_dir / "queries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


Q1 = {"id": "q1", "text": "rule following benchmark", "type": "lookup", "author": "human"}
Q2 = {"id": "q2", "text": "how good are RAG pipelines at time travel", "type": "no-answer",
      "author": "llm-draft"}


def test_load_builds_a_pdf_dataset_with_queries_and_qrels(fetched, cache):
    _write_queries(fetched, Q1, Q2)
    (fetched / "qrels.tsv").write_text("q1\t2311.04235\t2\nq1\t2104.08663\t0\n")
    ds = load(cache, data_dir=fetched)
    assert ds.name == "papers" and ds.doc_suffix == ".pdf"
    assert ds.docs["2311.04235"] == _pdf("rules v2")
    assert ds.titles["2104.08663"].startswith("BEIR:")
    assert ds.queries == {"q1": "rule following benchmark", "q2": Q2["text"]}
    assert ds.query_meta["q2"] == {"type": "no-answer", "author": "llm-draft"}
    assert ds.qrels == {"q1": {"2311.04235": 2, "2104.08663": 0}}


def test_load_without_queries_or_qrels_is_an_empty_eval(fetched, cache):
    ds = load(cache, data_dir=fetched)
    assert ds.queries == {} and ds.qrels == {} and len(ds.docs) == 2


def test_load_fetches_missing_pdfs_at_their_pinned_versions(fetched, cache, monkeypatch):
    (cache / "papers" / "2311.04235v2.pdf").unlink()
    arxiv = FakeArxiv(TABLE)
    monkeypatch.setattr(papers, "_http_get", lambda: arxiv)
    monkeypatch.setattr(papers.time, "sleep", _no_sleep)
    ds = load(cache, data_dir=fetched)
    assert ds.docs["2311.04235"] == _pdf("rules v2")
    assert arxiv.pdf_requests() == ["https://export.arxiv.org/pdf/2311.04235v2"]


def test_paper_missing_from_the_lock_must_be_fetched_first(fetched, cache):
    (fetched / "papers.tsv").write_text(
        (fetched / "papers.tsv").read_text() + "2210.07316\tMTEB\n")
    with pytest.raises(ValueError, match="2210.07316.*python -m evals.datasets.papers"):
        load(cache, data_dir=fetched)


@pytest.mark.parametrize("rows,qrels,match", [
    ([{**Q1, "type": "lookups"}], "", "line 1.*type"),
    ([{**Q1, "author": "me"}], "", "line 1.*author"),
    ([Q1, {**Q2, "id": "q1"}], "", "line 2.*q1"),
    ([{**Q1, "text": "  "}], "", "line 1.*text"),
    ([Q1], "q9\t2311.04235\t1\n", "qrels.tsv line 1.*q9"),
    ([Q1], "q1\t2210.07316\t1\n", "qrels.tsv line 1.*2210.07316"),
    ([Q1], "q1\t2311.04235\t3\n", "qrels.tsv line 1.*grade"),
    ([Q1], "q1\t2311.04235\t1\nq1\t2311.04235\t2\n", "qrels.tsv line 2.*twice"),
])
def test_load_rejects_bad_queries_and_qrels(fetched, cache, rows, qrels, match):
    _write_queries(fetched, *rows)
    if qrels:
        (fetched / "qrels.tsv").write_text(qrels)
    with pytest.raises(ValueError, match=match):
        load(cache, data_dir=fetched)


# ------------------------------------------------------- the committed set

LOCK = papers.DATA_DIR / "papers.lock.json"
needs_lock = pytest.mark.skipif(not LOCK.exists(), reason="papers set not fetched yet")


@needs_lock
def test_every_committed_label_matches_its_paper():
    labels = papers.read_papers(papers.DATA_DIR / "papers.tsv")
    lock = json.loads(LOCK.read_text())
    assert set(labels) == set(lock)
    bad = [i for i, label in labels.items()
           if not label_matches(label, lock[i]["title"], lock[i]["authors"])]
    assert bad == []


@needs_lock
def test_no_committed_label_matches_a_different_paper():
    labels = papers.read_papers(papers.DATA_DIR / "papers.tsv")
    lock = json.loads(LOCK.read_text())
    clashes = [(labels[i], j) for i in labels for j in lock
               if j != i and label_matches(labels[i], lock[j]["title"], lock[j]["authors"])]
    assert clashes == []


# ------------------------------------------------------------ review fixes

def test_update_can_repin_one_paper_and_leave_the_rest(data_dir, cache):
    fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep)
    newer = {
        "2311.04235": TABLE["2311.04235"] + [
            {"title": "Can LLMs Follow Simple Rules?", "authors": ["Norman Mu"], "pdf": _pdf("rules v3")}],
        "2104.08663": TABLE["2104.08663"] + [{**TABLE["2104.08663"][0], "pdf": _pdf("beir v2")}],
    }
    lock = fetch(data_dir, cache, http_get=FakeArxiv(newer), sleep=_no_sleep, update=["2311.04235"])
    assert lock["2311.04235"]["version"] == 3
    assert lock["2104.08663"]["version"] == 1


def test_update_of_a_paper_not_in_the_set_is_refused(data_dir, cache):
    with pytest.raises(ValueError, match="2210.07316"):
        fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep, update=["2210.07316"])


def test_sha_mismatch_advice_names_a_single_paper_update(data_dir, cache):
    fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep)
    (cache / "papers" / "2104.08663v1.pdf").unlink()
    regenerated = {**TABLE, "2104.08663": [{**TABLE["2104.08663"][0], "pdf": _pdf("re-rendered")}]}
    with pytest.raises(FetchError) as err:
        fetch(data_dir, cache, http_get=FakeArxiv(regenerated), sleep=_no_sleep)
    assert "--update 2104.08663" in err.value.problems[0]


def test_truncated_download_is_neither_cached_nor_pinned(data_dir, cache):
    cut = {**TABLE, "2104.08663": [{**TABLE["2104.08663"][0], "pdf": _pdf("beir v1")[:-7]}]}
    with pytest.raises(FetchError) as err:
        fetch(data_dir, cache, http_get=FakeArxiv(cut), sleep=_no_sleep)
    assert any("2104.08663" in p and "truncated" in p for p in err.value.problems)
    assert not (cache / "papers" / "2104.08663v1.pdf").exists()
    assert not (data_dir / "papers.lock.json").exists()


def test_an_interrupted_write_leaves_no_pdf_behind(data_dir, cache, monkeypatch):
    def interrupted(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(papers.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        fetch(data_dir, cache, http_get=FakeArxiv(TABLE), sleep=_no_sleep)
    assert not list((cache / "papers").glob("*.pdf"))


def test_loading_never_rewrites_the_lock(fetched, cache, monkeypatch):
    before = (fetched / "papers.lock.json").read_bytes()
    (cache / "papers" / "2311.04235v2.pdf").unlink()
    # arXiv's author metadata drifted; the pinned version, title and bytes didn't
    drifted = {**TABLE, "2311.04235": [
        {**v, "authors": ["Norman Mu", "Sarah Chen"]} for v in TABLE["2311.04235"]]}
    monkeypatch.setattr(papers, "_http_get", lambda: FakeArxiv(drifted))
    monkeypatch.setattr(papers.time, "sleep", _no_sleep)
    load(cache, data_dir=fetched)
    assert (fetched / "papers.lock.json").read_bytes() == before


@pytest.mark.parametrize("qid", ["q1 ", " q1", "q\t1", "q/1", "-q1"])
def test_query_ids_are_plain_names(fetched, cache, qid):
    _write_queries(fetched, {**Q1, "id": qid})
    with pytest.raises(ValueError, match="line 1.*id"):
        load(cache, data_dir=fetched)


def test_title_lookups_ask_for_each_paper_by_its_title():
    from evals.datasets import Dataset
    from evals.datasets.papers import title_lookups

    papers = Dataset(name="papers", docs={"2401.1": b"%PDF a", "2312.9": b"%PDF b"}, queries={"q01": "x"},
                     qrels={"q01": {"2401.1": 1}}, doc_suffix=".pdf", pooled=True,
                     titles={"2401.1": "Later Paper", "2312.9": "Earlier Paper"})
    ds = title_lookups(papers)
    assert ds.name == "papers-titles" and ds.docs == papers.docs and ds.doc_suffix == ".pdf"
    assert ds.queries == {"t01": "Earlier Paper", "t02": "Later Paper"}      # by arXiv id
    assert ds.qrels == {"t01": {"2312.9": 2}, "t02": {"2401.1": 2}}
    assert not ds.pooled            # each query's only relevant paper is known
