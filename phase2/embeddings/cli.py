"""CLI commands for the embedding pipeline."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from phase2.embeddings.config import EmbeddingEngineConfig, load_embedding_config
from phase2.embeddings.engine import EmbeddingEngine

console = Console()
embeddings_app = typer.Typer(
    name="embeddings",
    help="Phase 2 — Embedding Infrastructure",
    no_args_is_help=True,
)


def _get_engine(config_path: str) -> EmbeddingEngine:
    cfg = load_embedding_config(config_path)
    return EmbeddingEngine(cfg)


@embeddings_app.command()
def generate(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Recompute embeddings even if cached.",
    ),
) -> None:
    """Generate embeddings for all configured sources."""
    console.print("[bold cyan]Phase 2 — Embedding Generation[/bold cyan]")
    engine = _get_engine(config)
    result = engine.generate(force=force)

    table = Table(title="Embedding Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Input Rows", str(result["total_input"]))
    table.add_row("Embedded", str(result["total_embedded"]))
    table.add_row("Skipped (cached)", str(result["total_skipped"]))
    table.add_row("Errors", str(result["total_errors"]))
    table.add_row("Elapsed Time", f"{result['elapsed_seconds']}s")
    console.print(table)

    for src, info in result["by_source"].items():
        if isinstance(info, dict) and "status" in info:
            console.print(f"  [yellow]{src}: {info['status']} ({info.get('reason', '')})[/yellow]")
        elif isinstance(info, dict):
            console.print(f"  [green]{src}: {info['embedded']}/{info['total']}[/green]")


@embeddings_app.command()
def stats(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Display embedding statistics."""
    engine = _get_engine(config)
    s = engine.stats()

    table = Table(title="Embedding Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Vectors", str(s.total_vectors))
    table.add_row("Dimension", str(s.dimension))
    table.add_row("Mean Norm", f"{s.mean_norm:.6f}")
    table.add_row("Std Norm", f"{s.std_norm:.6f}")
    table.add_row("Null Text Snippets", str(s.null_text_snippets))
    console.print(table)

    if s.by_source:
        stable = Table(title="Per Source")
        stable.add_column("Source", style="cyan")
        stable.add_column("Count", style="green")
        for src, count in sorted(s.by_source.items()):
            stable.add_row(src, str(count))
        console.print(stable)


@embeddings_app.command()
def search(
    query: str = typer.Argument(..., help="Text query to search for."),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    k: int = typer.Option(
        10,
        "--top-k",
        "-k",
        help="Number of nearest neighbours.",
    ),
) -> None:
    """Search embeddings by cosine similarity."""
    engine = _get_engine(config)
    results = engine.search(query, k=k)

    if not results:
        console.print("[yellow]No embeddings found. Run 'generate' first.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Top {len(results)} Results for: '{query[:60]}'")
    table.add_column("#", style="dim")
    table.add_column("Source", style="cyan")
    table.add_column("Similarity", style="green")
    table.add_column("Source ID", style="yellow")
    table.add_column("Snippet", style="white")

    for i, r in enumerate(results, 1):
        snippet = (r.text_snippet or "")[:80]
        table.add_row(str(i), r.source_type.value, f"{r.similarity:.4f}", r.source_id[:12], snippet)

    console.print(table)


@embeddings_app.command()
def verify(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Verify integrity of stored embeddings."""
    engine = _get_engine(config)
    result = engine.verify()

    table = Table(title="Embedding Verification")
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="green")
    for check, status in result["checks"].items():
        color = "green" if status.startswith("PASS") else "red" if status.startswith("FAIL") else "yellow"
        table.add_row(check, f"[{color}]{status}[/{color}]")
    console.print(table)

    if result.get("total_vectors"):
        console.print(f"\nTotal vectors: {result['total_vectors']}")
        console.print(f"Dimension: {result['dimension']}")


if __name__ == "__main__":
    embeddings_app()