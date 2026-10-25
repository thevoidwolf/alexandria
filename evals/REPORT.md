# Alexandria retrieval report

Work in progress. Results, analyses and the failure log are added as each
phase lands; see [README.md](README.md) for how to run the harness.

## Hypotheses (pre-registered)

Written on 2026-10-23, before any eval run. They are recorded here unedited
so the results can be checked against them rather than the other way round.
Each will be reported as **supported**, **not supported**, or
**inconclusive** (the 95% CI crosses zero). Being wrong about one is fine
and worth saying.

- **H1:** `hybrid` ≥ both `fts` and `vec` on nDCG@10, on both datasets.
- **H2:** `hybrid+qi` gives a small, positive change.
- **H3:** `hybrid+rerank` improves nDCG@10 the most, at a large p95 latency cost.
- **H4:** hits with `matched_in = both` are more precise than single-retriever hits.
- **H5:** smaller chunks improve the papers set.
- **H6:** for queries of 5+ words, `hybrid-or` beats `hybrid` on nDCG@10.

### Further predictions from reading the code

Also made before any run, while reading `search.py` and `ingest/`:

- **P1:** FTS joins query words with an implicit AND (both on the first,
  raw-syntax attempt and in the `_quote_terms` fallback), and the FTS5
  tokenizer keeps stopwords. SciFact queries are full-sentence claims, so
  `fts` will return nothing for a large share of them and score well below
  the published BM25 numbers. Where FTS returns nothing, `hybrid` ranks
  exactly as `vec` does, so `hybrid` will sit close to `vec` on SciFact.
- **P2:** chunks are 800 *words*, but `bge-small-en-v1.5` truncates its
  input at 512 tokens, so the vector side only sees roughly the first
  400 words of each full-size chunk. SciFact abstracts are short, so this
  should matter little there; on the papers set it should favour smaller
  chunks (H5).

## Changes to the plan

Recorded before any papers-set results exist.

- **2026-10-23: `hybrid-c400` is replaced by `hybrid-c300`** (300-word
  chunks, 50-word overlap), on Voidwolf's call: a 400-word chunk is still
  about 630 bge tokens on SciFact's text, over the 512-token window, so it
  wouldn't test chunks that fit. `hybrid-c200` (200/40) stays. The 50-word
  overlap for c300 (17% of the chunk) was chosen to sit between c400's
  12.5% and c200's 20%.
  - *Measured afterwards, on the papers set:* its text averages 1.77 bge
    tokens per word, so 300-word chunks still overflow 512 tokens 37.6% of
    the time (200-word chunks 8.3%). Reported to Voidwolf; c300 stands unless
    he changes it, and any change is recorded here before papers results.
  - *Scripted later* (`evals.tokens`, from the committed indexes): 37.3% and
    7.9%. The one-off counts were taken before the PDF upload path was
    pinned (`bb3ed39`), the likeliest reason for the difference; see
    [What the embedding model sees](#what-the-embedding-model-sees-p2).
- **2026-10-23: the `fts-or` / `hybrid-or` definition is frozen** as used in
  a scratch SciFact run on the same day, and won't change before the papers
  results:
  - split the query on whitespace; drop words containing no letter or
    digit; wrap each remaining word in double quotes (doubling any `"`
    inside it); join them with ` OR `;
  - `fts-or` passes that expression to FTS5 as the MATCH query;
  - `hybrid-or` uses it on the FTS side only; the vector side embeds the
    original query; fusion (RRF, k = 60) and candidate depth are as in
    `hybrid`;
  - a query with no remaining words gets no FTS results.

  Disclosure: that scratch run is the only OR formulation tried, and its
  SciFact score (nDCG@10 0.686 for `fts-or`) was seen before this
  definition was frozen. The same scratch run tried `vec` with BGE's query
  instruction (0.713, below plain `vec`'s 0.720). Neither number has
  tuned anything. Both are re-measured as committed variants in phase 4.
- **2026-10-23: each `lookup` and `paraphrase` has its target paper graded
  2 before judging** (31 grades in `qrels.tsv`), on Voidwolf's call. Pooled
  judging only grades papers some variant returned. So a query whose
  target every variant missed, and which turned up nothing else relevant,
  would have no relevant papers and be left out of every mean. The misses
  that matter most would vanish instead of scoring 0. Those 31 pairs aren't
  graded blind; they are relevant because the query was written for them.
- **2026-10-23: eight `paraphrase` queries are LLM-drafted**, marked
  `llm-draft`. Voidwolf's notes had four paraphrases against a target of
  about twelve. A mix also matches how Alexandria is used, since agents
  query it as well as people.
  - *Papers:* the 20 with no `lookup` or `paraphrase` written for them,
    sorted by arXiv id, sampled with `random.Random(0).sample(pool, 8)`:
    BEIR, ALCE, C-Pack, SWE-bench, Control Illusion, When Reject Turns into
    Accept, Health-ORSC-Bench, Passing the Test You Trained On.
  - *Drafter:* a fresh Claude Opus 5.5 subagent with none of this project's
    context: not the hypotheses, the search code or the other queries. It
    got a file of each paper's title and abstract, taken from the pinned
    PDF, and this prompt (the file's path elided):

    > You are helping build a test set for a personal search tool. Someone
    > keeps a private library of research-paper PDFs and searches it by
    > typing queries. We need search queries to test it with.
    >
    > Read the one file `<titles and abstracts>` with the Read tool. It
    > holds 8 papers, each as `[arXiv id] title` followed by the abstract.
    > Use no other tools and no other sources: don't search, don't open
    > other files, don't look anything up.
    >
    > For each paper, write one query of this type:
    >
    >   paraphrase: one paper's idea in words different from the paper's.
    >
    > Write it the way the library's owner would type it into the search
    > box: someone who read the paper a while ago, remembers the idea, and
    > wants to find it again. One query per paper, 8 in all.
    >
    > Reply with only 8 lines of JSON, one per paper, in the order the file
    > lists them, like:
    > {"paper": "2104.08663", "text": "..."}

    Its eight lines are used unedited.
  - *Caveat:* all eight drafts are 17 to 22 words long, while the human
    queries run from 1 to 13 words. Any human-against-LLM comparison is
    therefore also a comparison of lengths and of papers, and is reported
    as description only.
- **2026-10-23: two of Voidwolf's notes were corrected against the pinned
  papers' text** before any run, by reading the papers, not by searching
  them. The RuLES lookup said 15 scenarios; the pinned v3 says 14. A ReAct
  lookup's placeholder became the name from the paper's worked example,
  "Arthur's Magazine was started in 1844".
- **2026-10-23: latency is timed after an untimed pass over every query**,
  replacing a single warm-up query. With one warm-up query, the variant run
  first in a process was slower: on the papers set `fts` took 3.0 ms at p50
  when run first and 0.4 ms when run last, and `vec` took 16.5 ms against
  14.2 to 14.8 ms. All committed latencies are re-measured this way;
  relevance results don't change.
- **2026-10-23: the papers summary reports `judged@10`** for each variant:
  the share of its top-10 papers that have a grade. A variant is scored
  only after its new pairs are graded, so this should read 100% in every
  reported result.
- **2026-10-23: the reranker's settings.** `hybrid+rerank` uses
  `BAAI/bge-reranker-base` on CPU over the top 50 fused chunks, the brief's
  defaults. The cross-encoder reads at most 512 tokens of query and chunk
  together, so it truncates long chunks, as the embedder does (P2).

## Decisions before the phase-4 grades

Recorded on 2026-10-25, after every variant had run on both sets and
before any of the 472 (query, paper) pairs new to the papers pool were
graded. By then the SciFact results for every variant had been seen, as
had the papers results for the three graded baselines and the other
variants' papers scores with their ungraded papers counted as not relevant
(`judged@10` 51–84%), which say little about those variants yet. The
phase-4 grades themselves had not been seen.

- **The rest runs on a second machine.** The machine that produced the
  committed rows is unavailable for about a week, so the remaining phases
  run on another (Ryzen 7 5800X, WSL2). Committed rows are not re-run: the
  judge reads passages by their stored chunk positions, and `--rescore`
  scores rows without searching, so rankings and latencies stay those of
  the first machine. The four papers indexes rebuilt here have the same
  cache keys as those the rows came from, which both tools check, and
  re-running all twelve papers variants here into a scratch folder
  reproduced every committed row's top 10, chunk positions and retriever
  ranks exactly. Any new timing is reported separately, with its hardware.
- **43 queries, not 50.** The brief targeted about 50. The set stays at
  43, on Voidwolf's call: every type is covered, with 12 paraphrases (the
  target) and 5 `no-answer`. Seven more queries would narrow the
  confidence intervals by about 7% (√(43/50) ≈ 0.93), but would mean
  re-running every papers variant and another round of grading, and they
  would be written knowing where search fails.
- **q42 keeps its type and its grade.** It was written as `no-answer`, and
  the blind judging pass graded one paper 1 for it (Health-ORSC-Bench,
  2601.17642). Neither changes now: the type was set before any run and the
  grade was given blind. The no-answer analysis counts it as answerable,
  with a note. A question written to have no answer that turns out to have
  a partial one is a small finding about writing such questions.
- **c300 stands.** No change to it was recorded before the papers results,
  as its entry above required.
- **Decision rules, set before the grades they apply to.** The brief
  leaves three product calls to Voidwolf until the results are in. Each
  gets its rule now, judged on the papers set by the nDCG@10 difference
  from `hybrid`, with the paired bootstrap 95% CI used throughout. None of
  the changes is made on this branch; each would be its own pull request.
  - *Reranking in the MCP tools.* Only if `hybrid+rerank`'s CI lies
    entirely above zero. Then it becomes an opt-in `rerank` parameter on
    `search_tool`, off by default, once its latency has been measured on a
    GPU. Otherwise it stays CLI-only, and the README says why.
  - *OR in keyword search.* The candidate is a fallback rather than either
    measured variant: run the query with AND, as now, and only if that
    finds nothing, run the OR expression frozen above. Where AND matches,
    this is exactly `hybrid`; where AND finds nothing, it is exactly
    `hybrid-or` (same FTS list, vector list and fusion). So it is scored
    from those two variants' rows without a new run: `hybrid-or`'s row for
    each query where `hybrid` had no FTS hits (`n_fts` = 0), `hybrid`'s row
    otherwise. Its latency comes from the same rows, which leaves out the
    failed AND attempt (under a millisecond on SciFact). It becomes the
    default if its CI's lower bound is above −0.02. It fixes a behaviour,
    natural-language queries silently falling back to vector-only search
    (P1), so it needs to show it costs nothing, not that it gains. The same
    change would stop reading an uppercase `OR`, `AND` or `NOT` in ordinary
    text as an FTS5 operator.

    *Disclosure:* the fallback is exploratory, not one of the
    pre-registered variants, and its SciFact score was computed and seen
    before this rule was set: nDCG@10 0.7327, +0.0108 [−0.0154, +0.0363]
    against `hybrid`; on queries of 5+ words, +0.0120 [−0.0137, +0.0373].
    H6 is still reported as registered, on `hybrid-or`.
  - *Chunk size.* The default changes only if `hybrid-c200` or
    `hybrid-c300` has a CI entirely above zero. If one does, chunks are
    sized by the embedding model's tokens rather than by words, since even
    300-word chunks overflow its 512-token window 37.6% of the time on this
    set, and the change comes with a way to re-index an existing corpus.

## Results

The tables between `<!-- report:… -->` markers are generated from
`results/` by `uv run python -m evals.report`, and the text around them is
written by hand. On the papers set, a table that depends on grades stays
empty until every variant's top 10 is judged, because each new grade can
move every variant's score. Latencies are warm, per `search()` call, on the
machine that produced the rows (Ryzen 7 5700X3D, 16 threads, WSL2, CPU
only). Δ is the difference from `hybrid`, the default install, with a
paired bootstrap 95% CI, in bold where the CI excludes zero.

### SciFact

Every variant has run and the judgements are BEIR's, so these results are
final.

<!-- report:main scifact -->
| Variant | nDCG@10 | Recall@5 | Recall@10 | MRR@10 | Δ nDCG@10 [95% CI] | p50 / p95 ms |
|---|---|---|---|---|---|---|
| `fts` | 0.0467 | 0.0458 | 0.0458 | 0.0500 | **−0.675 [−0.720, −0.630]** | 0.5 / 0.9 |
| `vec` | 0.7200 | 0.7653 | 0.8452 | 0.6845 | −0.002 [−0.006, +0.001] | 20.0 / 23.4 |
| `hybrid` | 0.7219 | 0.7686 | 0.8452 | 0.6872 | baseline | 21.0 / 24.3 |
| `hybrid+qi` | 0.7158 | 0.7607 | 0.8362 | 0.6876 | −0.006 [−0.019, +0.007] | 21.7 / 25.6 |
| `hybrid+rerank` | 0.7191 | 0.7682 | 0.8434 | 0.6873 | −0.003 [−0.031, +0.026] | 11,940 / 13,147 |
| `vec-base` | 0.7390 | 0.8221 | 0.8659 | 0.7019 | +0.017 [−0.005, +0.040] | 41.7 / 49.3 |
| `hybrid-base` | 0.7411 | 0.8212 | 0.8659 | 0.7053 | +0.019 [−0.003, +0.042] | 42.2 / 49.7 |
| `hybrid-k20` | 0.7219 | 0.7686 | 0.8452 | 0.6872 | +0.000 [+0.000, +0.000] | 19.8 / 23.7 |
| `fts-or` | 0.6863 | 0.7543 | 0.8249 | 0.6482 | **−0.036 [−0.071, −0.002]** | 22.3 / 38.6 |
| `hybrid-or` | 0.7319 | 0.7876 | 0.8629 | 0.6959 | +0.010 [−0.016, +0.036] | 39.5 / 58.0 |
<!-- /report -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/ndcg-scifact-dark.svg">
  <img src="results/charts/ndcg-scifact.svg" alt="SciFact: each variant's nDCG@10 difference from hybrid with its 95% CI. Only fts and fts-or are clear of zero, both below it; every other CI crosses zero.">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/latency-scifact-dark.svg">
  <img src="results/charts/latency-scifact.svg" alt="SciFact: nDCG@10 against latency per query. Most variants sit between 0.716 and 0.741 at 20 to 50 ms; hybrid+rerank scores the same at about 12 seconds; fts is sub-millisecond at 0.047.">
</picture>

- **On SciFact's claims, the default install is vector search.** FTS's
  implicit AND finds nothing for 284 of the 300 queries, and on all 284
  `hybrid`'s top 10 is exactly `vec`'s, so the two are level (P1).
- **Two comparisons the [README](README.md#against-published-numbers)
  left open are now measured.** With its words joined by OR, FTS5's BM25
  scores 0.686, 0.007 above Anserini's flat BM25 (0.679): `fts`'s 0.047 is
  the query, not the ranking function. With BGE's query instruction,
  `hybrid+qi` (on SciFact, nearly vector search with the instruction)
  scores 0.716 against the published bge-small number with it (0.713), so
  most of the 0.007 gap the README noted between `vec` and that number is
  the instruction. It doesn't help: without it, `hybrid` scores 0.722.
- **Nothing beats `hybrid` with a CI clear of zero.** `hybrid-base` comes
  closest (+0.019 [−0.003, +0.042]), at twice the latency and three times
  the build time.
- **Reranking changes nothing here and costs about 12 s a query** on CPU
  (−0.003 [−0.031, +0.026]).
- **RRF k = 20 changes nothing:** `hybrid-k20` returns `hybrid`'s top 10
  for all 300 queries, the 16 FTS matched included.
- **OR alone loses to `hybrid`** (`fts-or`, −0.036 [−0.071, −0.002]), and
  OR nearly doubles the latency of a hybrid search (`hybrid-or`, 39.5 ms
  at p50 against 21.0 ms): FTS now matches almost every query, and has to
  rank what it matches.

### Papers set

<!-- report:main papers -->
_Waiting for the phase-4 grades._
<!-- /report -->

### Cost

<!-- report:cost scifact -->
| Index (model, chunk/overlap) | Variants | Documents | Chunks | Build (s) | Database (MB) |
|---|---|---|---|---|---|
| bge-small, 800/100 words | `fts`, `vec`, `hybrid`, `hybrid+qi`, `hybrid+rerank`, `hybrid-k20`, `fts-or`, `hybrid-or` | 5,183 | 5,192 | 274 | 63.2 |
| bge-base, 800/100 words | `vec-base`, `hybrid-base` | 5,183 | 5,192 | 816 | 72.6 |

Built on AMD Ryzen 7 5700X3D 8-Core Processor, embedding on CPU. A build is timed once, and build times move with machine load.
<!-- /report -->

<!-- report:cost papers -->
| Index (model, chunk/overlap) | Variants | Documents | Chunks | Build (s) | Database (MB) |
|---|---|---|---|---|---|
| bge-small, 800/100 words | `fts`, `vec`, `hybrid`, `hybrid+qi`, `hybrid+rerank`, `hybrid-k20`, `fts-or`, `hybrid-or` | 37 | 595 | 76 | 17.8 |
| bge-base, 800/100 words | `vec-base`, `hybrid-base` | 37 | 595 | 174 | 19.4 |
| bge-small, 300/50 words | `hybrid-c300` | 37 | 1,640 | 160 | 22.6 |
| bge-small, 200/40 words | `hybrid-c200` | 37 | 2,558 | 224 | 23.3 |

Built on AMD Ryzen 7 5700X3D 8-Core Processor, embedding on CPU. A build is timed once, and build times move with machine load.
<!-- /report -->

### What the embedding model sees (P2)

Counted by `uv run python -m evals.tokens` over every chunk in each index,
with the embedding model's own tokenizer.

SciFact:

<!-- report:tokens scifact -->
Extracted text averages 1.56 tokens per word with bge-small-en-v1.5's tokenizer, and 8.8% of the 5,183 documents are longer than its 512-token window ([CLS] and [SEP] included).

| Chunk / overlap (words) | Chunks | Over 512 tokens | Median tokens | 95th percentile |
|---|---|---|---|---|
| 800/100 | 5,192 | 8.8% | 316 | 568 |
<!-- /report -->

Papers set:

<!-- report:tokens papers -->
Extracted text averages 1.77 tokens per word with bge-small-en-v1.5's tokenizer, and all 37 documents are longer than its 512-token window ([CLS] and [SEP] included).

| Chunk / overlap (words) | Chunks | Over 512 tokens | Median tokens | 95th percentile |
|---|---|---|---|---|
| 800/100 | 595 | 98.7% | 1,269 | 2,068 |
| 300/50 | 1,640 | 37.3% | 475 | 828 |
| 200/40 | 2,558 | 7.9% | 313 | 559 |
<!-- /report -->

P2's premise holds on the papers set: almost every default-size chunk is
longer than the window, so the vector side reads roughly the first 290
words of an 800-word chunk (510 tokens at 1.77 a word) and none of the
rest. On SciFact it hardly matters, as predicted. Whether the papers set
favours smaller chunks is H5.

These counts replace one-off ones made while planning. SciFact's 1.57
tokens per word, in the README until now, counted [CLS] and [SEP] as part
of each abstract; counted without them it is 1.56. The papers set's
37.6% and 8.3% (recorded in [Changes to the plan](#changes-to-the-plan))
are 37.3% and 7.9% here. The 1.77 tokens per word and the 99% overflow at
800 words are unchanged. None of the differences changes a decision.

### Hypotheses

Each is judged per dataset. The papers set's verdicts come with its
grades.

- **H1** (`hybrid` ≥ `fts` and `vec`). SciFact: well above `fts` (+0.675
  [+0.630, +0.720]) and level with `vec` (+0.002 [−0.001, +0.006]), so
  **inconclusive**: the CI against `vec` crosses zero, as P1 predicted.
- **H2** (`hybrid+qi` gives a small, positive change). SciFact: −0.006
  [−0.019, +0.007], **inconclusive**, with a negative point estimate.
- **H3** (`hybrid+rerank` improves nDCG@10 the most, at a large p95
  latency cost). SciFact: **inconclusive** on the gain (−0.003 [−0.031,
  +0.026]), and not the largest point estimate (`hybrid-base`'s is). The
  cost is real: p95 13.1 s against 24.3 ms.
- **H4** (`matched_in = both` hits are more precise). SciFact:
  **supported** for `hybrid`, on few hits; see below.
- **H5** (smaller chunks improve the papers set). Papers set only.
- **H6** (for queries of 5+ words, `hybrid-or` beats `hybrid`). SciFact:
  +0.010 [−0.016, +0.036] on 297 queries, **inconclusive**.
- **P1** is confirmed on SciFact: FTS finds nothing for 95% of queries,
  `fts` scores 0.047 against the published BM25's 0.679, and `hybrid` is
  level with `vec`.
- **P2**: see above.

### Is a hit found by both retrievers more likely relevant? (H4)

The comment on `SearchHit.matched_in` in `search.py` says "a hit in both is
materially more confident than a hit in either alone". Rank is a
confounder: fusion puts documents both retrievers found near the top, and
top ranks are more often relevant anyway. So precision is also split by
rank, and the gap between the two kinds of hit gets a CI.

SciFact:

<!-- report:matched-in scifact -->
| Rows | Best chunk found by | Precision, top 10 | Ranks 1–3 | Ranks 4–10 |
|---|---|---|---|---|
| `hybrid` | both | 0.79 (15/19) | 0.79 (15/19) | — |
| `hybrid` | FTS only | — | — | — |
| `hybrid` | vector only | 0.09 (271/2981) | 0.25 (217/881) | 0.03 (54/2100) |
| `hybrid`, queries FTS matched | both | 0.79 (15/19) | 0.79 (15/19) | — |
| `hybrid`, queries FTS matched | FTS only | — | — | — |
| `hybrid`, queries FTS matched | vector only | 0.03 (4/141) | 0.07 (2/29) | 0.02 (2/112) |
| `hybrid-or` | both | 0.10 (289/2994) | 0.26 (238/900) | 0.02 (51/2094) |
| `hybrid-or` | FTS only | 0.00 (0/3) | — | 0.00 (0/3) |
| `hybrid-or` | vector only | 0.00 (0/3) | — | 0.00 (0/3) |

Precision of documents found by both minus those found by one, with a 95% CI from resampling queries (— where either kind has fewer than 10 hits):

| Rows | Ranks 1–3 | Ranks 4–10 |
|---|---|---|
| `hybrid` | **+0.543 [+0.340, +0.754]** | — |
| `hybrid`, queries FTS matched | **+0.721 [+0.483, +0.941]** | — |
| `hybrid-or` | — | — |

Precision: the share of returned documents graded relevant (1 or 2), with the count; unjudged documents count as not relevant. Each document counts once, by the retrievers that found its best chunk among the 300 candidates each fetched.
<!-- /report -->

On SciFact the claim holds for the default search. A document both found
is relevant 15 times in 19, all at ranks 1–3, against 217 in 881 for
vector-only hits at those ranks. The gap is wider on the 16 queries FTS
matched, where vector-only hits beside the FTS match are rarely relevant.
Two limits. First, "both" only happens when FTS's AND matches, which marks
the short, specific queries that are easy anyway. Second, with OR the
label stops meaning anything: 2,994 of `hybrid-or`'s 3,000 top-10 hits are
found by both. Whether the comment needs qualifying waits for the papers
set.

Papers set:

<!-- report:matched-in papers -->
_Waiting for the phase-4 grades._
<!-- /report -->

### By query type (papers set)

<!-- report:by-type papers -->
_Waiting for the phase-4 grades._
<!-- /report -->

The eight `llm-draft` paraphrases against the four `human` ones,
described rather than tested: they differ in length and in paper as well
as in author (see [Changes to the plan](#changes-to-the-plan)).

<!-- report:by-author papers -->
_Waiting for the phase-4 grades._
<!-- /report -->

### Queries with nothing relevant (papers set)

<!-- report:no-answer papers -->
_Waiting for the phase-4 grades._
<!-- /report -->

### FTS finding nothing, and OR (H6)

How often the default AND finds nothing, by query length. It uses no
grades, so both sets are complete.

SciFact:

<!-- report:zero-hit scifact -->
| Query length (words) | Queries | AND (default) finds nothing | OR finds nothing |
|---|---|---|---|
| 1–4 | 3 | 2 of 3 (67%) | 0 of 3 (0%) |
| 5–8 | 68 | 59 of 68 (87%) | 0 of 68 (0%) |
| 9–12 | 100 | 95 of 100 (95%) | 0 of 100 (0%) |
| 13+ | 129 | 128 of 129 (99%) | 0 of 129 (0%) |
| all | 300 | 284 of 300 (95%) | 0 of 300 (0%) |
<!-- /report -->

Papers set:

<!-- report:zero-hit papers -->
| Query length (words) | Queries | AND (default) finds nothing | OR finds nothing |
|---|---|---|---|
| 1–4 | 16 | 2 of 16 (12%) | 0 of 16 (0%) |
| 5–8 | 11 | 4 of 11 (36%) | 0 of 11 (0%) |
| 9–12 | 7 | 5 of 7 (71%) | 0 of 7 (0%) |
| 13+ | 9 | 9 of 9 (100%) | 0 of 9 (0%) |
| all | 43 | 20 of 43 (47%) | 0 of 43 (0%) |
<!-- /report -->

`hybrid-or` against `hybrid`, and the AND-then-OR fallback from the
decision rules, scored from the same rows:

<!-- report:or scifact -->
| Compared with `hybrid` | Queries | n | `hybrid` nDCG@10 | nDCG@10 | Δ [95% CI] |
|---|---|---|---|---|---|
| `hybrid-or` | all | 300 | 0.7219 | 0.7319 | +0.010 [−0.016, +0.036] |
| `hybrid-or` | 5+ words | 297 | 0.7267 | 0.7366 | +0.010 [−0.016, +0.036] |
| `hybrid-or` | 1–4 words | 3 | 0.2487 | 0.2725 | +0.024 [−0.431, +0.387] |
| AND, then OR if AND finds nothing | all | 300 | 0.7219 | 0.7327 | +0.011 [−0.015, +0.036] |
| AND, then OR if AND finds nothing | 5+ words | 297 | 0.7267 | 0.7387 | +0.012 [−0.014, +0.037] |
| AND, then OR if AND finds nothing | 1–4 words | 3 | 0.2487 | 0.1436 | −0.105 [−0.431, +0.115] |

The fallback runs OR for 284 of 300 (95%) queries. Its latency, from the same rows (so without the failed AND attempt): p50 39.4 ms, p95 58.0 ms.
<!-- /report -->

<!-- report:or papers -->
_Waiting for the phase-4 grades._
<!-- /report -->

### Decision rules

Applied as set in [Decisions before the phase-4
grades](#decisions-before-the-phase-4-grades); what Voidwolf decides is
recorded here once the papers set is graded.

<!-- report:rules papers -->
_Waiting for the phase-4 grades._
<!-- /report -->

### Failure log

The ten worst `hybrid` queries on the papers set, written up by hand once
the grades are in.
