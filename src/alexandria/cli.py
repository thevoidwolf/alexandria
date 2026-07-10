from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

from alexandria.config import load as load_config
from alexandria.db import connect
from alexandria.ingest import ingest_file, ingest_folder, ingest_url
from alexandria.ingest.fetch import RobotsDisallowed
from alexandria.search import search

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

VALID_MODES = {"hybrid", "fts", "vec"}


def _split_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _friendly_errors(fn):
    """Turn expected exceptions into typer.Exit(1) with a red error line."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as e:
            console.print(f"[red]error:[/] file not found: {e}")
        except NotADirectoryError as e:
            console.print(f"[red]error:[/] not a directory: {e}")
        except RobotsDisallowed as e:
            console.print(f"[red]error:[/] robots.txt disallows fetching {e}")
        except httpx.HTTPError as e:
            console.print(f"[red]error:[/] HTTP failure: {e}")
        except ValueError as e:
            console.print(f"[red]error:[/] {e}")
        raise typer.Exit(code=1)

    return wrapper


@app.command()
@_friendly_errors
def ingest(
    path: Path = typer.Argument(..., help="File to ingest"),
    category: Optional[str] = typer.Option(None, "--category", "-c"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated"),
) -> None:
    """Ingest a single file end-to-end."""
    cfg = load_config()
    conn = connect(cfg.db_path, cfg.embeddings.dim)
    result = ingest_file(path, conn, cfg, category=category, tags=_split_tags(tags))

    if result.was_duplicate:
        console.print(
            f"[yellow]duplicate[/] via [bold]{result.dedup_gate}[/] gate → doc_id=[cyan]{result.doc_id}[/]"
        )
    else:
        console.print(
            f"[green]ingested[/] doc_id=[cyan]{result.doc_id}[/] "
            f"type={result.content_type} chunks={result.n_chunks} "
            f"title={result.title!r}"
        )


@app.command("ingest-folder")
@_friendly_errors
def ingest_folder_cmd(
    path: Path = typer.Argument(..., help="Folder to ingest"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R"),
    glob: Optional[str] = typer.Option(None, "--glob", "-g", help="Filename glob"),
    category: Optional[str] = typer.Option(None, "--category", "-c"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated"),
) -> None:
    """Walk a folder and ingest supported files."""
    cfg = load_config()
    conn = connect(cfg.db_path, cfg.embeddings.dim)
    result = ingest_folder(
        path, conn, cfg,
        recursive=recursive, glob=glob,
        category=category, tags=_split_tags(tags),
    )
    console.print(
        f"[green]{len(result.ingested)}[/] new · "
        f"[yellow]{len(result.duplicates)}[/] duplicates · "
        f"[red]{len(result.errors)}[/] errors · "
        f"{result.scanned} scanned"
    )
    for path_str, reason in result.errors:
        console.print(f"  [red]![/] {path_str} — {reason}")


@app.command("ingest-url")
@_friendly_errors
def ingest_url_cmd(
    url: str = typer.Argument(..., help="URL to fetch and ingest"),
    category: Optional[str] = typer.Option(None, "--category", "-c"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated"),
) -> None:
    """Fetch a URL and ingest its content."""
    cfg = load_config()
    conn = connect(cfg.db_path, cfg.embeddings.dim)
    result = ingest_url(url, conn, cfg, category=category, tags=_split_tags(tags))

    if result.was_duplicate:
        console.print(
            f"[yellow]duplicate[/] via [bold]{result.dedup_gate}[/] gate → "
            f"doc_id=[cyan]{result.doc_id}[/]"
        )
    else:
        console.print(
            f"[green]ingested[/] doc_id=[cyan]{result.doc_id}[/] "
            f"type={result.content_type} chunks={result.n_chunks} "
            f"title={result.title!r}"
        )


@app.command("list")
def list_cmd(
    category: Optional[str] = typer.Option(None, "--category", "-c"),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """List ingested documents."""
    cfg = load_config()
    conn = connect(cfg.db_path, cfg.embeddings.dim)

    q = (
        "SELECT id, content_type, category, title, bytes, ingested_at "
        "FROM documents"
    )
    args: list = []
    if category:
        q += " WHERE category = ?"
        args.append(category)
    q += " ORDER BY ingested_at DESC LIMIT ?"
    args.append(limit)

    rows = conn.execute(q, args).fetchall()

    table = Table(show_header=True, header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("type")
    table.add_column("category")
    table.add_column("title")
    table.add_column("bytes", justify="right")
    table.add_column("ingested_at")

    for r in rows:
        table.add_row(
            r[0], r[1], r[2] or "", (r[3] or "")[:60], str(r[4]), r[5]
        )
    console.print(table)
    console.print(f"[dim]{len(rows)} row(s)[/]")


@app.command("search")
@_friendly_errors
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    category: Optional[str] = typer.Option(None, "--category", "-c"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated (AND)"),
    limit: int = typer.Option(10, "--limit", "-n"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="hybrid | fts | vec"),
) -> None:
    """Search the corpus (hybrid FTS + vector by default)."""
    if mode not in VALID_MODES:
        raise typer.BadParameter(
            f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}"
        )
    cfg = load_config()
    conn = connect(cfg.db_path, cfg.embeddings.dim)
    hits = search(
        query, conn, cfg,
        category=category, tags=_split_tags(tags),
        limit=limit, mode=mode,  # type: ignore[arg-type]
    )

    if not hits:
        console.print("[dim]no matches[/]")
        return

    table = Table(show_header=True, header_style="bold", show_lines=True)
    table.add_column("score", justify="right", style="green")
    table.add_column("type")
    table.add_column("category")
    table.add_column("title")
    table.add_column("snippet")

    for h in hits:
        table.add_row(
            f"{h.score:.4f}",
            h.content_type,
            h.category or "",
            (h.title or "")[:40],
            h.snippet[:120],
        )
    console.print(table)


@app.command()
def info() -> None:
    """Show data directory and embedding model."""
    cfg = load_config()
    console.print(f"home:  {cfg.home}")
    console.print(f"db:    {cfg.db_path}")
    console.print(f"model: {cfg.embeddings.model} (dim={cfg.embeddings.dim})")


@app.command()
def mcp() -> None:
    """Run the MCP stdio server."""
    from alexandria.server import run
    run()


if __name__ == "__main__":
    app()
