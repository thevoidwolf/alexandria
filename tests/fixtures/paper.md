# A short note on hybrid retrieval

Modern search stacks combine lexical retrieval, typically BM25 over an inverted
index, with dense retrieval, typically nearest-neighbour search over vector
embeddings. Reciprocal rank fusion is a simple, tuning-free method to combine
the two rankings without needing calibrated scores from either side. Alternatives
include weighted sums with min-max normalization and learned rerankers such as
monoBERT and monoT5.

Dense retrievers excel at matching paraphrases and synonyms — "how to fix my
laptop" retrieving documents about "repairing computers" — but they are weak on
rare tokens like invoice numbers, chemical formulas, and version strings, where
BM25 shines.

## Filters and metadata

In a personal knowledge store like Alexandria, filters over category and tag
metadata are essential to keep unrelated documents from crowding the top of
results. Research papers should not surface when the user is looking through
household bills.
