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
