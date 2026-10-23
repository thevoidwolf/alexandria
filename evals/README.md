# Retrieval evals

A harness that measures how well Alexandria's search finds the right
documents, and what each retrieval choice costs. Findings, hypotheses and
the failure log are in [REPORT.md](REPORT.md).

## Running it

```bash
uv sync --group eval            # pyarrow + ir-measures; not needed by Alexandria itself
uv run python -m evals.run --dataset scifact --variants all
uv run python -m evals.crosscheck --dataset scifact

uv run python -m evals.datasets.papers          # fetch, check and pin the arXiv set
uv run python -m evals.run --dataset papers --variants all
uv run python -m evals.judge                    # grade pooled results (papers set)
```

The first SciFact run downloads the corpus (about 5 MB) and builds an index
from its 5,183 abstracts, which takes a few minutes on CPU. Later runs reuse the
index. `--variants` takes `all` or a comma-separated list (`fts,vec`).

Results go to `evals/results/<dataset>/`:

- `<variant>.jsonl`: one row per query: `query_id`, `variant`, `mode`,
  `n_words`, `latency_ms`, `docs` (top 10 doc ids, best first), `hits` (for
  each of those docs, its best chunk's position in the document (`chunk`),
  `fts_rank`, `vec_rank` and `matched_in`), `n_hits`/`n_fts`/`n_vec` (how many of the 100 returned chunks
  each retriever found), and the four metrics.
- `summary.json`: per-variant means, latency, index cost (including the
  embedding model's Hub revision), the FTS zero-hit rate, differences from
  the baseline with confidence intervals, and the environment that produced
  each variant's rows (CPU, library versions, commit). Running a subset of
  variants updates their entries and keeps the others.

On the same hardware, rerunning from a fresh clone reproduces the committed
rows exactly, apart from `latency_ms`: checked by rebuilding the SciFact
index from an empty cache. A different CPU or BLAS library could swap the
order of near-tied vector hits. Alexandria has no setting to pin the
embedding model, so it loads whatever is on the Hub's `main`; the revision
it got is recorded with each index and is part of the cache key.

## Isolation

Every index is a throwaway Alexandria home under
`evals/.cache/<dataset>-<hash>/home`, opened by explicit path, so the eval
never reads or writes your own corpus. Documents go in through
`ingest_fetched`, the same call `store_document_tool` makes, so extraction,
normalization, chunking and embedding are all part of what's measured.

The hash covers the corpus, the embedding model and its Hub revision, the
chunk settings and the source of `alexandria/ingest/` and `db.py`. Changing
any of them builds a new index; an ingest fix can't be measured against a
stale one. A build
that was interrupted (no `build.json`) is discarded and redone.

## How results are scored

**Chunks to documents.** Alexandria returns chunks; relevance judgements are
per document. Each query runs `search(query, limit=100)`, the chunks are
collapsed to documents in order of first appearance, and the top 10
documents are scored. This is a choice, not a given: a document whose best
chunk ranks 11th still scores at rank 2 if only one other document comes
before it. On SciFact almost every abstract is a single chunk, so the
choice barely matters there.

`limit=100` also widens what `hybrid` fuses: `search()` takes
`max(3 × limit, 30)` candidates from each retriever, so 300 here against 30
for `search_tool`'s default `limit=10`. On SciFact that changes the top 10
for 5 of 300 queries and leaves mean nDCG@10 the same, but it matters more
when FTS returns many results (the `-or` variants), so read `hybrid` as
"the default install's ranking with a wider candidate window".

**Metrics.** They follow trec_eval's conventions, which is what BEIR's
published numbers use, and are cross-checked against
[ir-measures](https://github.com/terrierteam/ir_measures) (unit tests on
random rankings, plus `python -m evals.crosscheck` after a full run).
For the committed SciFact run (commit `2a5fcbe`) all three variants matched
pytrec_eval: largest difference 2.2e-16 over 3,600 per-query values.

| Metric | Meaning |
|---|---|
| nDCG@10 | Headline metric, comparable with BEIR. Rewards finding relevant docs *and* ranking them high. Linear gain, log2 discount, ideal ranking cut at 10. |
| Recall@5, Recall@10 | Share of the relevant docs that came back at all. What matters most to an agent reading the results. |
| MRR@10 | 1 / rank of the first relevant doc (0 if none in the top 10). |
| Latency p50 / p95 | Wall time of one `search()` call: query embedding plus both retrievers plus fusion. Warm: one untimed query first loads the model. Hardware is in `summary.json`. |
| Index build time, DB size | The cost side. Build time is the whole ingest of the corpus. |

- **A query that returns nothing scores 0.** It's a miss, not a gap, and it
  stays in the mean.
- **A query with no relevant documents has no score.** It is left out of
  that metric's mean (the `n` beside each mean says how many counted).
  trec_eval would instead score it 0 and keep it in the mean; leaving it
  out keeps unanswerable queries from dragging every variant down equally.
  SciFact has none; the papers set will have about five.
- **Unjudged documents count as non-relevant.**
- **Differences from the baseline** (`hybrid`, the default install) are the
  mean per-query difference with a paired bootstrap 95% CI: 10,000
  resamples of the queries, seed 0. A difference whose CI crosses zero is
  not called real.
- **Identical documents.** Alexandria stores identical texts once, so two
  dataset docs can become one stored document. Both ids map to it, and the
  judgements are merged at the higher grade. Results use dataset ids
  (Alexandria's ids change on every build). The count is `n_collapsed` in
  `summary.json`.

## Datasets

### BEIR SciFact

[SciFact](https://huggingface.co/datasets/BeIR/scifact) (CC BY-SA 4.0):
5,183 paper abstracts and, in the test split, 300 scientific claims with 339
relevance judgements (all grade 1). Downloaded from the Hugging Face Hub at
pinned revisions (`BeIR/scifact@b3b5335`, `BeIR/scifact-qrels@2938d17`);
nothing from it is committed. Each abstract is stored as a `.txt` document
whose body is `title + "\n\n" + text`, the same "flat" layout as Anserini's
BM25 baseline below.

SciFact is the sanity check: a public benchmark with published numbers to
compare against. It is small and its documents are short.

#### Baselines

From `evals/results/scifact/` (CPU: Ryzen 7 5700X3D, 16 threads, WSL2;
full environment in `summary.json`). Δ is against `hybrid`, the default
install, with a paired bootstrap 95% CI.

| Variant | nDCG@10 | Recall@5 | Recall@10 | MRR@10 | Δ nDCG@10 [95% CI] | p50 / p95 ms |
|---|---|---|---|---|---|---|
| `fts` | 0.0467 | 0.0458 | 0.0458 | 0.0500 | −0.675 [−0.720, −0.630] | 0.5 / 0.9 |
| `vec` | 0.7200 | 0.7653 | 0.8452 | 0.6845 | −0.002 [−0.006, +0.001] | 18.6 / 22.0 |
| `hybrid` | 0.7219 | 0.7686 | 0.8452 | 0.6872 | baseline | 19.4 / 23.5 |

The index holds 5,183 documents in 5,192 chunks (7 abstracts run past 800
words; none collapsed as duplicates). Built in 274 s on CPU (306 s on an
earlier build of the same index: build time varies with machine load);
the database is 63 MB.

#### Against published numbers

| System | nDCG@10 | Source |
|---|---|---|
| BM25, Anserini, "flat" (title + text in one field, as here) | 0.6789 | [Anserini regression](https://github.com/castorini/anserini/blob/59d6fa720ffdf2af7f68e453ca97fa5b452289bf/docs/reproduce/from-document-collection/beir-v1.0.0-scifact.flat.md) |
| BM25, Anserini, multifield (title and text weighted equally) | 0.6647 | [Anserini regression](https://github.com/castorini/anserini/blob/59d6fa720ffdf2af7f68e453ca97fa5b452289bf/docs/reproduce/from-document-collection/beir-v1.0.0-scifact.multifield.md) |
| bge-small-en-v1.5, with BGE's query instruction | 0.7128 | [Model card](https://huggingface.co/BAAI/bge-small-en-v1.5) (MTEB SciFact, `ndcg_at_10` 71.275); instruction per [FlagEmbedding's eval script](https://github.com/FlagOpen/FlagEmbedding/blob/fd1a2bdf69488ffebe0327999d4400d8c8058a0b/research/C_MTEB/eval_MTEB.py) |

**`fts` scores 0.047 against BM25's 0.679, and almost all of that gap is
how the query is built, not BM25.** `search.py` hands the query to FTS5 as
written, and FTS5 joins bare words with an implicit AND; the
`_quote_terms` fallback (used when the query isn't valid FTS5 syntax)
does the same. A chunk only matches if it contains every word of the
query, stopwords included, since the `porter unicode61` tokenizer keeps
them. SciFact queries are whole-sentence claims, so FTS returns nothing for
284 of the 300 (94.7%), and the miss rate climbs with length (from the
committed rows):

| Query length | FTS finds nothing |
|---|---|
| 1–4 words | 2 of 3 |
| 5–8 words | 59 of 68 (87%) |
| 9–12 words | 95 of 100 (95%) |
| 13+ words | 128 of 129 (99%) |

On the 16 queries where FTS does match, it scores nDCG@10 0.875. That
isn't proof of a working BM25 on its own: these are the queries short or
specific enough for every word to appear in one abstract, and they are
easy for vectors too (`vec` scores 0.895 on them). Anserini's query is a disjunction (any term may
match, ranked by BM25), so the like-for-like comparison is the `fts-or`
variant (phase 4). Smaller differences that remain after that: FTS5's
`bm25()` uses k1 = 1.2, b = 0.75 where Anserini uses 0.9 and 0.4; Lucene's
English analyzer drops stopwords and FTS5's tokenizer doesn't; and
Alexandria scores chunks, not documents (identical here, since almost
every abstract is a single chunk).

**`vec` scores 0.720 against the published 0.713.** The biggest known
difference: BGE's MTEB evaluation prepends "Represent this sentence for
searching relevant passages: " to every query, and Alexandria's default
(`embeddings.query_instruction = ""`) doesn't. The `hybrid+qi` variant
(phase 4) measures that directly. Smaller ones: BGE's evaluation joins
title and text with a space where we use a blank line
([`flag_dres_model.py`](https://github.com/FlagOpen/FlagEmbedding/blob/fd1a2bdf69488ffebe0327999d4400d8c8058a0b/research/C_MTEB/flag_dres_model.py));
Alexandria NFKC-normalizes the text before embedding; and the 7 abstracts
longer than 800 words are split into chunks and scored by their best one.
Both truncate at 512 tokens, so truncation isn't a difference between
them.

**`hybrid` is statistically indistinguishable from `vec`**, because RRF
over one non-empty list just reproduces that list. On all 284 queries
where FTS found nothing, hybrid's top 10 is exactly vec's. The difference
is confined to the 16 queries FTS matched, where hybrid scores 0.931
against vec's 0.895 — too few to conclude anything, but the direction
hybrid is supposed to go. In other words, on natural-language queries
the default install is vector search; see [REPORT.md](REPORT.md)
prediction P1.

### arXiv papers set

37 public arXiv papers on agents, evals, retrieval and instruction-following,
listed in [`data/papers/papers.tsv`](data/papers/papers.tsv) as
`arxiv_id <TAB> label`. Each paper is ingested from its PDF through the web
upload path, with the PDF backend pinned to `pypdf`, so text extraction is
part of what's measured. The PDFs are fetched into `evals/.cache/papers/`
and never committed: arXiv's default licence doesn't allow redistribution.

**Fetching is checked and pinned.** `python -m evals.datasets.papers` asks
the arXiv API (`export.arxiv.org`, one request every 3 s) about every id and
stops, listing every problem and leaving the lock alone, if:

- an id doesn't resolve on arXiv;
- a label doesn't match its paper's title;
- a pinned paper's title, or its PDF's sha256, no longer matches
  [`papers.lock.json`](data/papers/papers.lock.json), which records the
  version, exact title, authors and hash of each paper.

Later fetches use the pinned versions. If arXiv re-renders a PDF (same
version, different bytes), check it and re-pin just that paper with
`--update <id>`; a bare `--update` re-pins every paper, which changes the
corpus every committed result was measured on.

Labels are short names, not titles, so "matches" is defined: every word of
the label, or of a note in parentheses, appears in the title, ignoring
case, punctuation and a few stopwords; "LLM(s)" and "RAG" may match their
spelled-out forms; a note like "(Lewis et al.)" instead requires that author.
A data test checks the committed set both ways: every label matches its own
paper and no other paper's title. Matching against abstracts was tried and
rejected, because abstracts name other benchmarks: 10 wrong pairings passed.

**What extraction produces.** The default index holds 37 papers in 595
chunks (5 to 33 per paper), built in 75 s on CPU, 18 MB. pypdf found a
`/Title` for only 12 of the 37 PDFs, the known title-fallback issue. Search
doesn't use titles, but `judge.py` shows the arXiv title instead. With the
model's tokenizer the extracted text averages **1.77 tokens per word**
(references, math and URLs tokenize badly; SciFact's abstracts average
1.57). So 99% of the default 800-word chunks overflow bge-small's 512-token
window, and even 300-word chunks overflow it 37.6% of the time (median 475
tokens, 95th percentile 833); 200-word chunks overflow 8.3% of the time.
These are one-off measurements, to be scripted with the report.

#### Queries and judgements

The queries are Voidwolf's. Write them, one JSON object per line, in
`data/papers/queries.jsonl`:

```json
{"id": "q01", "text": "FormatSpread", "type": "lookup", "author": "human"}
```

- `type` is one of:
  - `lookup`: exact names, numbers or identifiers.
  - `paraphrase`: one paper's idea in words different from the paper's.
  - `conceptual`: an idea across papers.
  - `no-answer`: something the set can't answer.
- `author` is `human`, or `llm-draft` for a query an LLM drafted. LLM-drafted
  queries tend to reuse the source's wording, which favours keyword search,
  so they are reported separately.

A suggested mix for 50: 15 `lookup`, 15 `paraphrase`, 15 `conceptual`, 5
`no-answer`, with lengths spread from one or two words to full sentences
(the AND-versus-OR comparison and the zero-hit-by-length analysis need both).
Write them before looking at what search returns for them.

Judgements live in `data/papers/qrels.tsv` (`query_id <TAB> arxiv_id <TAB>
grade`), written by `judge.py`:

1. `uv run python -m evals.run --dataset papers --variants all` runs every
   variant, so there is something to pool. Its scores are blank until
   there are judgements.
2. `uv run python -m evals.judge` pools each query's top 10 papers from
   every variant run so far and shuffles them. For each unjudged pair it
   shows the query, the paper's title and an 80-word stretch of a
   retrieved passage (`m` for 200 words of it). It never shows the variant,
   the rank, or how many variants found the paper, and the 200-word cap is
   the smallest chunk size any variant uses, so a passage's length can't
   give a chunk-size variant away. Grade 2 if the paper is
   what the query is looking for, 1 if it addresses the query, 0 if not:
   the grade is for the paper, and the passage is evidence. Every grade is
   saved as it's given, so quit (`q`) whenever; the next session resumes.
   Run one session at a time.
3. Run `evals.run` again to score.

To fix a mis-keyed grade, delete its line from `qrels.tsv` and run the judge
again; it asks for that pair again. Rewording a query after its results
were produced makes the judge refuse them until `evals.run` is re-run, and
rewording one that's already judged would leave its old grades attached:
give it a new id instead.

When phase 4 adds variants, steps 1–3 repeat and only the new pairs are
asked. Results from an index other than the current one are refused,
because their passages would be wrong. With the three baselines, queries
in two smoke tests pooled 10 to 12 papers each (FTS adds a few beyond what
vector search returns, mostly for short lookups), so 50 queries would be
roughly 500 to 600 grades; every new variant adds the papers only it found.

Papers no variant returned are never judged and count as not relevant.
That biases scores in two ways. Recall is overstated, because relevant
papers no variant found are missing from its denominator. And a variant
scored before its own new pairs are judged (a newly added variant, or a
judging session stopped halfway) has those papers counted as not relevant.
The second goes away by judging before scoring.

The loader checks both files on every run and names the line of any unknown
type or author, duplicate id, empty query, judgement for an unknown query
or paper, grade outside 0–2, or pair judged twice.

### Known caveats

- **Embedding truncation.** `bge-small-en-v1.5` reads at most 512 tokens.
  Counted with its tokenizer over the SciFact corpus (a one-off
  measurement, not yet scripted; phase 5's report will be), a word averages
  1.57 tokens, so 512 tokens is about
  326 words, while a default chunk is 800 words (~1,257 tokens): the
  vector side never sees roughly the last 60% of a full-size chunk. Only
  8.8% of SciFact abstracts exceed 512 tokens, so this matters little
  here; it will matter on the papers set (REPORT.md P2).
- **Latency.** Embedding the query takes about 12 ms (p50, CPU, measured
  separately) of the ~19 ms `vec` and `hybrid` take; the rest is the
  vector scan and loading the 100 hits. `fts` is sub-millisecond, but on
  SciFact that is mostly the cost of finding nothing.
- **One machine.** All numbers come from a single CPU under WSL2.
