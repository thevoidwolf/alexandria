from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingsConfig:
    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    dim: int = 384


@dataclass(frozen=True)
class ChunkingConfig:
    tokens: int = 800
    overlap: int = 100


@dataclass(frozen=True)
class StorageConfig:
    keep_blobs: bool = True


@dataclass(frozen=True)
class HttpConfig:
    user_agent: str = "Alexandria/0.1 (+local knowledge store)"
    respect_robots_txt: bool = True
    max_redirects: int = 5
    timeout_seconds: int = 30


@dataclass(frozen=True)
class CategoryRule:
    host_glob: str
    category: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PdfExtractorConfig:
    backend: str = "pypdf"                # "pypdf" | "marker" | "auto"
    device: str = "auto"                  # "auto" | "cuda" | "cpu" | "mps"
    math_symbol_threshold: float = 2.0    # per 1000 chars; consulted only in "auto"
                                          # Prose with occasional Greek hits ~1.5;
                                          # 2.0 leaves margin. Font-name signal
                                          # catches LaTeX math regardless of density.

    # Thermal safety knobs for marker (M10). Only affect marker code paths.
    marker_batch_size: int = 0            # 0 = surya defaults (device-picked);
                                          # 1 = serial (safest on thermal-constrained
                                          # GPUs like the RTX A1000); 2-4 balances.
    marker_cooldown_seconds: float = 0.0  # sleep after each marker run; helps
                                          # folder ingest by letting the card cool
                                          # between docs. 0 = no cooldown.


@dataclass(frozen=True)
class ExtractorsConfig:
    pdf: PdfExtractorConfig = field(default_factory=PdfExtractorConfig)


@dataclass(frozen=True)
class NetworkConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    auth_token_file: str | None = None      # explicit path; env/CLI can override
    allowed_hosts: tuple[str, ...] = ()      # extra Host: header values MCP will accept
                                             # (network-visible IPs, MagicDNS names,
                                             # reverse-proxy hostnames). Loopback is
                                             # always allowed.
    public_base_url: str | None = None       # e.g. "https://alexandria.suki.tail-net.ts".
                                             # Used to build shareable signed-URL
                                             # responses for get_original. When unset,
                                             # falls back to "http://<host>:<port>" only
                                             # if host is loopback.


@dataclass(frozen=True)
class WebConfig:
    enabled: bool = True
    title: str = "Alexandria"
    max_upload_mb: int = 100
    password_hash_file: str | None = None    # default: $HOME/web_password_hash
    session_secret_file: str | None = None   # default: $HOME/session_secret
    session_max_age_days: int = 30


@dataclass(frozen=True)
class ClassifyConfig:
    """Knobs for suggest_metadata (nearest-neighbor tag/category propagation).

    Suggestions come from the k nearest documents by chunk-embedding
    similarity. A tag or category is only suggested when it appears on a
    large-enough fraction of those neighbors — high fractions favor
    precision (fewer, more confident suggestions), low fractions favor
    recall.
    """
    tag_min_fraction: float = 0.35        # tag must be on ≥ this fraction of neighbors
    category_min_fraction: float = 0.35   # same for category (majority pick)
    max_tags: int = 7                     # cap suggestion at this many tags
    # Cosine-similarity floor for an anchor to count. Categories have
    # a lower bar than tags because a doc *must* have a category — a
    # weak-match top pick is still the best-available answer. Tags are
    # additive and optional, so a weak match is worse than no tag; the
    # bar is higher to keep spillover (adjacent-but-absent topics
    # scoring in the 0.5s) out of the tag set.
    anchor_min_similarity_category: float = 0.45
    anchor_min_similarity_tag: float = 0.60
    # When True, after each ingest the classifier runs and attaches a
    # suggestion to the job's result JSON. Never auto-applies. Skips
    # duplicates and docs where the user supplied category/tags at ingest.
    # The home page reads this from the job SSE payload and offers
    # one-click apply beneath the row; empty-corpus cases fall back to
    # anchor-only matches and are safely skipped when nothing clears
    # the threshold.
    suggest_on_ingest: bool = True


@dataclass(frozen=True)
class Config:
    home: Path
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    extractors: ExtractorsConfig = field(default_factory=ExtractorsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    web: WebConfig = field(default_factory=WebConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    category_rules: tuple[CategoryRule, ...] = ()

    @property
    def db_path(self) -> Path:
        return self.home / "alexandria.db"

    @property
    def blobs_dir(self) -> Path:
        return self.home / "blobs"

    @property
    def pending_uploads_dir(self) -> Path:
        return self.blobs_dir / "pending"

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"

    @property
    def web_password_hash_path(self) -> Path:
        return Path(self.web.password_hash_file).expanduser() if self.web.password_hash_file \
            else self.home / "web_password_hash"

    @property
    def web_session_secret_path(self) -> Path:
        return Path(self.web.session_secret_file).expanduser() if self.web.session_secret_file \
            else self.home / "session_secret"


def _default_home() -> Path:
    if env := os.environ.get("ALEXANDRIA_HOME"):
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "alexandria"


def load(home: Path | None = None) -> Config:
    home = (home or _default_home()).resolve()
    home.mkdir(parents=True, exist_ok=True)
    (home / "blobs").mkdir(exist_ok=True)

    cfg_file = home / "config.toml"
    data: dict = {}
    if cfg_file.exists():
        with cfg_file.open("rb") as f:
            data = tomllib.load(f)

    emb = EmbeddingsConfig(**(data.get("embeddings") or {}))
    chunk = ChunkingConfig(**(data.get("chunking") or {}))
    storage = StorageConfig(**(data.get("storage") or {}))
    http = HttpConfig(**(data.get("http") or {}))

    extractors_raw = data.get("extractors") or {}
    extractors = ExtractorsConfig(
        pdf=PdfExtractorConfig(**(extractors_raw.get("pdf") or {})),
    )

    network_raw = dict(data.get("network") or {})
    if "allowed_hosts" in network_raw:
        network_raw["allowed_hosts"] = tuple(network_raw["allowed_hosts"] or ())
    network = NetworkConfig(**network_raw)

    web = WebConfig(**(data.get("web") or {}))

    classify = ClassifyConfig(**(data.get("classify") or {}))

    rules_raw = data.get("category_rules") or []
    rules = tuple(
        CategoryRule(
            host_glob=r["host_glob"],
            category=r["category"],
            tags=tuple(r.get("tags", [])),
        )
        for r in rules_raw
    )

    return Config(
        home=home,
        embeddings=emb,
        chunking=chunk,
        storage=storage,
        http=http,
        extractors=extractors,
        network=network,
        web=web,
        classify=classify,
        category_rules=rules,
    )
