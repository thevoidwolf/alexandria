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
