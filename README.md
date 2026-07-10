# Alexandria

A local, single-user knowledge store with an MCP interface. Ingests PDFs, HTML,
plain text and markdown from folders or URLs; deduplicates by content hash;
retrieves with hybrid keyword + semantic search.

See [SPEC.md](SPEC.md) for the design document.

## Install

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```sh
uv sync
```

The first ingest downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB)
into your HuggingFace cache.

## Data directory

Alexandria stores everything under `$XDG_DATA_HOME/alexandria/` (falling back to
`~/.local/share/alexandria/`). Override with `ALEXANDRIA_HOME`.

```
$ALEXANDRIA_HOME/
├── alexandria.db     # SQLite with FTS5 + sqlite-vec
├── blobs/            # original bytes, content-addressed by sha256
└── config.toml       # (optional) user overrides
```

## CLI

```sh
# ingest a single file with a category and tags
uv run alexandria ingest ~/Docs/paper.pdf --category research --tags ml,retrieval

# walk a folder (recursive, supported extensions only by default)
uv run alexandria ingest-folder ~/Docs/Bills --category bills --tags 2026

# fetch and ingest a URL; category may come from config.toml rules
uv run alexandria ingest-url https://arxiv.org/abs/2004.04906

# search (hybrid RRF fusion by default)
uv run alexandria search "reciprocal rank fusion" --limit 5
uv run alexandria search "8891-3320-2" --category bills --mode fts
uv run alexandria search "how do dense retrievers work" --mode vec

# browse the corpus
uv run alexandria list --category research
uv run alexandria info
```

## Config (optional)

`$ALEXANDRIA_HOME/config.toml` — everything has sensible defaults; you only need
to override what you care about.

```toml
[embeddings]
model = "BAAI/bge-small-en-v1.5"
device = "cpu"                   # "cpu" | "cuda" | "mps"

[chunking]
tokens = 800
overlap = 100

[storage]
keep_blobs = true                # keep original bytes for provenance

[extractors.pdf]
backend = "pypdf"                # "pypdf" | "marker"
device  = "auto"                 # "auto" | "cuda" | "cpu" | "mps"

[http]
user_agent = "Alexandria/0.1"
respect_robots_txt = true
max_redirects = 5
timeout_seconds = 30

# URL heuristics: applied to ingest-url when the caller doesn't supply a category.
# host_glob uses fnmatch (so *.example.com matches subdomains).
[[category_rules]]
host_glob = "arxiv.org"
category  = "research"
tags      = ["paper"]

[[category_rules]]
host_glob = "*.pge.com"
category  = "bills"
tags      = ["utility", "electric"]
```

## MCP integration

Alexandria exposes six tools over MCP stdio: `ingest_folder_tool`,
`ingest_url_tool`, `search_tool`, `get_document_tool`, `list_documents_tool`,
`get_catalog_tool`.

Add to your MCP client (e.g. `~/.claude/mcp.json` for Claude Code):

```json
{
  "mcpServers": {
    "alexandria": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/Alexandria", "alexandria", "mcp"]
    }
  }
}
```

## Development

```sh
uv sync                           # dev deps included in `dev` group
uv run pytest                     # ~9 s cold, 50 tests
uv run ruff check src tests
```

Tests use hash-based fake embeddings by default (see `tests/conftest.py::fake_embed`)
so ingest/dedup/catalog tests run fast. Search tests use the real embedding model
via `sentence-transformers`' local cache.

## PDF backends

Two extractors, selected per install via `[extractors.pdf] backend`:

- **pypdf** (default) — no extra deps, fast on CPU, but math/tables come out as
  flattened plain text (fractions split across lines, sub/superscripts lost). No
  OCR: scanned PDFs raise a "no extractable text" error.
- **marker** — Torch-based, GPU-accelerated. Outputs markdown with `$…$` and
  `$$…$$` LaTeX for math, `#`-headings for structure, and OCR (surya) on scanned
  pages. Downloads ~2GB of weights to `~/.cache/datalab/` on first run. CPU
  inference is impractical (tens of minutes per paper); a modest GPU (~8GB VRAM)
  is enough.

Install marker only where you'll run it (server-primary is the intended shape):

```sh
uv sync --extra marker            # torch + marker-pdf + surya
```

then in `$ALEXANDRIA_HOME/config.toml`:

```toml
[extractors.pdf]
backend = "marker"
device  = "auto"                 # picks cuda > mps > cpu
```

If marker raises during ingest, Alexandria logs a warning and falls back to
pypdf on that document so ingest stays robust.

## Deduplication

Three cheap layers, always run, in this order:

1. **Source** — same `(source_kind, source_uri)` was ingested before.
2. **Raw bytes** — `sha256(bytes)` matches an existing document.
3. **Normalized text** — `sha256(normalized_text)` matches (catches re-exported
   PDFs, mirrored HTML, format conversions).

A duplicate hit updates `last_seen_at`, records the alternate `source_uri`, and
merges any new tags. Near-duplicate detection (MinHash/simhash) is not implemented.
