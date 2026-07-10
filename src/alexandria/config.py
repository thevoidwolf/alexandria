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
    backend: str = "pypdf"    # "pypdf" | "marker"
    device: str = "auto"      # "auto" | "cuda" | "cpu" | "mps"


@dataclass(frozen=True)
class ExtractorsConfig:
    pdf: PdfExtractorConfig = field(default_factory=PdfExtractorConfig)


@dataclass(frozen=True)
class NetworkConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    auth_token_file: str | None = None      # explicit path; env/CLI can override


@dataclass(frozen=True)
class Config:
    home: Path
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    extractors: ExtractorsConfig = field(default_factory=ExtractorsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    category_rules: tuple[CategoryRule, ...] = ()

    @property
    def db_path(self) -> Path:
        return self.home / "alexandria.db"

    @property
    def blobs_dir(self) -> Path:
        return self.home / "blobs"

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"


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

    network = NetworkConfig(**(data.get("network") or {}))

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
        category_rules=rules,
    )
