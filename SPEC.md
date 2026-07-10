# Alexandria — Local Knowledge Store with MCP Interface

Status: **draft** · Owner: voidwolf · Last updated: 2026-07-10

## 1. Goals

- **Local-first, single-user.** One portable data directory, no external services required.
- **Ingest once, retrieve well.** Content-addressed deduplication; hybrid keyword + semantic search.
- **Mixed corpora.** Research papers and household bills coexist without one drowning out the other — categories and tags do the partitioning.
- **MCP-native.** Everything an MCP client (Claude Code, etc.) needs is exposed as tools.

Non-goals for v1: multi-user, remote sync, near-duplicate detection, LLM auto-classification, OCR of image-only PDFs.

## 2. Architecture

```
┌─────────────── MCP client (Claude Code, etc.) ───────────────┐
                              │
                        MCP stdio server
                              │
     ┌────────────────────────┼────────────────────────┐
     │                        │                        │
  Ingestion               Search API              Catalog API
     │                        │                        │
 Fetchers → Extractors → Normalizer → Hasher → Chunker → Embedder
                              │
                    ┌─────────┴─────────┐
                    │                   │
              SQLite (FTS5)     sqlite-vec (vectors)
              + metadata        (same DB file)
                    │
              blobs/  (original files, content-addressed by sha256)
```

**Data directory** — `$XDG_DATA_HOME/alexandria/`, falling back to `~/.local/share/alexandria/`. Overridable via `ALEXANDRIA_HOME` env var.

Contents:

- `alexandria.db` — SQLite with FTS5 + sqlite-vec extension. Metadata, full text, keyword index, and vector index all in one file.
- `blobs/<sha256[:2]>/<sha256>` — original file bytes, content-addressed. Kept for provenance and re-extraction.
- `config.toml` — user config (embedding model, chunk sizes, URL heuristics).

## 3. Ingestion pipeline

Each source (file or URL) flows through:

1. **Fetch** — get bytes + source metadata (path or URL, mtime, HTTP content-type).
2. **Content-type detect** — sniff magic bytes; classify as `pdf | html | txt | md`. Don't trust extensions blindly.
3. **Raw-bytes hash** (`sha256_raw`) → *dedup gate 1*. If seen, skip re-processing and return the existing `doc_id`.
4. **Extract** — text + structural metadata (title, author, publication date when available):
   - PDF: `pypdf` for text-first PDFs; fall back to `pdfplumber` if extracted text is empty.
   - HTML: `trafilatura` (main-content extraction, strips nav/ads/boilerplate).
   - TXT/MD: read as UTF-8; markdown is kept as-is (searchable as text).
5. **Normalize** — collapse whitespace, strip zero-widths, Unicode NFC.
6. **Text hash** (`sha256_text`) → *dedup gate 2*. Catches re-exported PDFs, mirrored HTML, and format conversions of the same content.
7. **Chunk** — 800-token windows with 100-token overlap. Store chunk char offsets back into the full text.
8. **Embed** — per-chunk vectors. Default model recorded on the document so we can re-embed selectively when swapping models.
9. **Index** — INSERT into `documents`, `chunks`, FTS5 mirrors, and `sqlite-vec` mirror in a single transaction.

Failure mode: on `ingest_folder`, a single bad file is skipped and reported in the `errors[]` return; the walk continues.

## 4. Deduplication

Three layers, each cheap enough to always run:

| Layer | Key | Catches |
|---|---|---|
| 1. Source | `(source_kind, source_uri)` | Same file/URL ingested twice |
| 2. Raw bytes | `sha256(raw_bytes)` | Same file under different names/paths |
| 3. Normalized text | `sha256(normalized_text)` | Same content, different container (PDF↔HTML, re-export) |

On a hit at any layer: return the existing `doc_id`, update `last_seen_at`, and append the new source to `document_sources` (a document can have many sources).

Near-duplicate detection (MinHash/simhash) is explicitly **out of scope for v1**.

## 5. Data model (SQLite)

```sql
documents (
  id            TEXT PRIMARY KEY,     -- ulid
  sha256_raw    TEXT NOT NULL UNIQUE,
  sha256_text   TEXT NOT NULL,        -- indexed, not unique
  content_type  TEXT NOT NULL,        -- pdf|html|txt|md
  title         TEXT,
  author        TEXT,
  published_at  TEXT,                 -- ISO8601 if known
  category      TEXT,                 -- single primary category
  extracted_text TEXT NOT NULL,       -- full normalized text
  extractor     TEXT NOT NULL,        -- e.g. 'pypdf@4.2'
  embed_model   TEXT NOT NULL,
  bytes         INTEGER NOT NULL,
  ingested_at   TEXT NOT NULL,
  last_seen_at  TEXT NOT NULL
);

document_sources (                    -- one doc, many sources
  doc_id       TEXT NOT NULL,
  source_kind  TEXT NOT NULL,         -- 'file' | 'url'
  source_uri   TEXT NOT NULL,
  seen_at      TEXT NOT NULL,
  PRIMARY KEY (doc_id, source_kind, source_uri)
);

tags (
  doc_id  TEXT NOT NULL,
  tag     TEXT NOT NULL,
  PRIMARY KEY (doc_id, tag)
);

chunks (
  id          TEXT PRIMARY KEY,
  doc_id      TEXT NOT NULL,
  ord         INTEGER NOT NULL,
  start_char  INTEGER,
  end_char    INTEGER,
  text        TEXT NOT NULL
);

-- Virtual tables
documents_fts  -- FTS5 over title + extracted_text
chunks_fts     -- FTS5 over chunks.text
chunks_vec     -- sqlite-vec (chunk_id → embedding)
```

## 6. Search

Hybrid retrieval per query:

1. **FTS5** top-N (BM25) over `chunks_fts`, filtered by category/tags if provided.
2. **Vector** top-N over `chunks_vec` with the same filter.
3. **Fuse** with reciprocal rank fusion (RRF, `k=60`). No tuning parameters needed.
4. **Rerank (optional, later).** A cross-encoder if higher precision is wanted. Off by default.

Returned rows: `{chunk_id, doc_id, score, snippet, title, category, tags, source_uri}`.

## 7. Categorization

Two mechanisms, both explicit — no LLM auto-classification in v1:

1. **User-supplied at ingest.** `category` (single string) and `tags` (list). When provided, these always win.
2. **URL/domain heuristics** as defaults for `ingest_url` when the caller doesn't supply a category. Rules live in `config.toml`:

   ```toml
   [[category_rules]]
   host_glob = "arxiv.org"
   category  = "research"
   tags      = ["paper"]

   [[category_rules]]
   host_glob = "*.pge.com"
   category  = "bills"
   tags      = ["utility", "electric"]
   ```

   Path-glob rules for `ingest_folder` are a symmetric future addition; not in v1.

## 8. MCP tools (v1 surface)

```
ingest_folder(path, recursive=true, category?, tags?, glob?)
  → {ingested: [doc_id...], skipped_duplicates: N, errors: [{path, reason}, ...]}

ingest_url(url, category?, tags?)
  → {doc_id, was_duplicate: bool, title, category, tags}

search(query, category?, tags?, limit=10, mode="hybrid")
  → [{chunk_id, doc_id, score, snippet, title, category, tags, source_uri}, ...]

get_document(doc_id, include_text=false)
  → {full metadata + optional extracted_text}

list_documents(category?, tags?, since?, limit=50, offset=0)
  → paged list of doc metadata (no text body)

get_catalog()
  → {categories: [{name, count}], tags: [{name, count}], total_docs, total_bytes}
```

## 9. Configuration

`config.toml` in the data directory. Defaults shown; user only needs to override what they care about.

```toml
[embeddings]
model = "BAAI/bge-small-en-v1.5"   # 384-dim, higher MTEB than MiniLM
device = "cpu"                      # "cpu" | "cuda" | "mps"

[chunking]
tokens = 800
overlap = 100

[storage]
keep_blobs = true                   # keep original bytes for provenance

[http]
user_agent = "Alexandria/0.1 (+local knowledge store)"
respect_robots_txt = true
max_redirects = 5
timeout_seconds = 30

# [[category_rules]] entries as shown in §7
```

## 10. Repository layout

```
alexandria/
  pyproject.toml            # uv-managed, python 3.12+
  SPEC.md                   # this file
  src/alexandria/
    __init__.py
    server.py               # MCP stdio server (mcp SDK)
    config.py               # load config.toml
    db.py                   # sqlite connection, migrations, sqlite-vec load
    ingest/
      __init__.py           # orchestrator
      fetch.py              # file + http fetchers
      extract.py            # pdf/html/txt/md extractors
      hash.py               # raw + text hashers
      chunk.py
      embed.py              # sentence-transformers wrapper
    search.py               # FTS5 + vec + RRF
    catalog.py              # list/get/facets
    categorize.py           # URL heuristics
  tests/
    fixtures/               # small sample pdf/html/txt files
    test_ingest.py
    test_dedup.py
    test_search.py
```

## 11. Resolved defaults (previously "open questions")

| Question | Decision | Rationale |
|---|---|---|
| Data dir location | XDG (`$XDG_DATA_HOME/alexandria/`), fallback `~/.local/share/alexandria/`, override via `ALEXANDRIA_HOME` | Linux-idiomatic; still trivially portable |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) | Higher MTEB than MiniLM at similar cost; same vector dimensionality keeps `sqlite-vec` schema stable |
| URL fetch politeness | Respect robots.txt (config-toggleable); custom UA string; follow up to 5 redirects; 30s timeout | Safe default for a personal tool that may hit small sites |
| Blob retention | Keep originals by default; `storage.keep_blobs = false` opt-out | Disk is cheap; provenance and re-extraction are valuable |
| `ingest_folder` failure mode | Skip-and-continue, report per-file errors in `errors[]` | One bad PDF should never abort a nightly ingest |
| Chunk size / overlap | 800 tokens / 100 overlap | Reasonable default for BGE-small; tunable in config |

## 12. Milestones

1. **M1 — Skeleton.** Repo, config loader, SQLite schema + migrations, sqlite-vec loaded, no MCP yet. CLI to ingest a single file end-to-end.
2. **M2 — Ingest.** All four content types, all three dedup layers, folder walker, URL fetcher with heuristics.
3. **M3 — Search.** FTS5 + vec + RRF, filters by category/tags.
4. **M4 — MCP.** Wire the six tools to the ingest/search/catalog layers. Manual test with Claude Code.
5. **M5 — Polish.** Fixture corpus + tests, README with install/config, error surfaces reviewed.
