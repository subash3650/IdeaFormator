"""CLI commands for the Semantic Relationship Engine."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from phase2.similarity.config import load_similarity_config
from phase2.similarity.engine import SimilarityEngine

console = Console()
similarity_app = typer.Typer(
    name="similarity",
    help="Phase 2 — Semantic Relationship Engine",
    no_args_is_help=True,
)


def _get_engine(config_path: str) -> SimilarityEngine:
    cfg = load_similarity_config(config_path)
    return SimilarityEngine(cfg)


@similarity_app.command()
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
        help="Recompute relationships even if existing.",
    ),
) -> None:
    """Generate semantic relationships from embeddings."""
    console.print("[bold cyan]Phase 2 \u2014 Semantic Relationship Generation[/bold cyan]")
    engine = _get_engine(config)
    result = engine.generate(force=force)

    table = Table(title="Relationship Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Relationships", str(result.get("total_relationships", 0)))
    if "unique_source_ids" in result:
        table.add_row("Unique Source IDs", str(result["unique_source_ids"]))
        table.add_row("Unique Target IDs", str(result["unique_target_ids"]))
        table.add_row("Unique Pairs", str(result["unique_pairs"]))
        table.add_row("Avg Similarity", f"{result['avg_similarity']:.6f}")
        table.add_row("Avg Confidence", f"{result['avg_confidence']:.6f}")
    if "avg_neighbors" in result and result["avg_neighbors"] > 0:
        table.add_row("Avg Neighbors", f"{result['avg_neighbors']:.2f}")
    if "density" in result and result["density"] > 0:
        table.add_row("Density", f"{result['density']:.8f}")
    if "threshold_warning" in result and result["threshold_warning"]:
        table.add_row("Threshold Warning", f"[yellow]{result['threshold_warning']}[/yellow]")
    if "elapsed_seconds" in result:
        table.add_row("Elapsed Time", f"{result['elapsed_seconds']}s")
    if "reason" in result:
        table.add_row("Status", result["reason"])
    console.print(table)


@similarity_app.command()
def stats(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Display relationship statistics."""
    engine = _get_engine(config)
    s = engine.detailed_stats()

    table = Table(title="Relationship Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Relationships", str(s.total_relationships))
    table.add_row("Unique Source IDs", str(s.unique_source_ids))
    table.add_row("Unique Target IDs", str(s.unique_target_ids))
    table.add_row("Unique Pairs", str(s.unique_pair_ids))
    table.add_row("Avg Similarity", f"{s.avg_similarity:.6f}")
    table.add_row("Avg Confidence", f"{s.avg_confidence:.6f}")
    table.add_row("Relationship Density", f"{s.density:.8f}")
    table.add_row("Average Neighbors", f"{s.average_neighbors:.2f}")
    table.add_row("Max Neighbors", str(s.max_neighbors))
    table.add_row("Min Neighbors", str(s.min_neighbors))
    table.add_row("Connected Nodes", str(s.connected_nodes))
    table.add_row("Isolated Nodes", str(s.isolated_nodes))
    console.print(table)

    if s.relationship_type_counts:
        rtable = Table(title="By Relationship Type")
        rtable.add_column("Type", style="cyan")
        rtable.add_column("Count", style="green")
        for rtype, count in sorted(s.relationship_type_counts.items()):
            rtable.add_row(rtype, str(count))
        console.print(rtable)

    if s.source_type_counts:
        stable = Table(title="By Source Type")
        stable.add_column("Source", style="cyan")
        stable.add_column("Count", style="green")
        for src, count in sorted(s.source_type_counts.items()):
            stable.add_row(src, str(count))
        console.print(stable)

    if s.target_type_counts:
        ttable = Table(title="By Target Type")
        ttable.add_column("Target", style="cyan")
        ttable.add_column("Count", style="green")
        for tgt, count in sorted(s.target_type_counts.items()):
            ttable.add_row(tgt, str(count))
        console.print(ttable)


@similarity_app.command()
def search(
    source_id: str = typer.Argument(..., help="Source ID to find relationships for."),
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
        help="Number of top relationships.",
    ),
) -> None:
    """Find relationships for a given source ID."""
    engine = _get_engine(config)
    results = engine.search_relationships(source_id, k=k)

    if not results:
        console.print("[yellow]No relationships found for this source ID.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Top {len(results)} Relationships for: {source_id[:40]}")
    table.add_column("#", style="dim")
    table.add_column("Target", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Similarity", style="green")
    table.add_column("Confidence", style="green")

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.target_id[:20],
            r.target_type.value,
            f"{r.similarity_score:.4f}",
            f"{r.confidence:.4f}",
        )

    console.print(table)


@similarity_app.command()
def verify(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Verify integrity of stored relationships."""
    engine = _get_engine(config)
    result = engine.verify()

    table = Table(title="Relationship Verification")
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="green")
    for check, status in result["checks"].items():
        color = "green" if status.startswith("PASS") else "red" if status.startswith("FAIL") else "yellow"
        table.add_row(check, f"[{color}]{status}[/{color}]")
    console.print(table)

    if result.get("total_relationships"):
        console.print(f"\nTotal relationships: {result['total_relationships']}")


if __name__ == "__main__":
    similarity_app()