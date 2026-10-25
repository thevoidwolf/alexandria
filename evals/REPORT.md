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
