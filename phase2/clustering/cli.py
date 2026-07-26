"""CLI commands for the Semantic Clustering Engine."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from phase2.clustering.config import load_clustering_config
from phase2.clustering.engine import ClusteringEngine

console = Console()
clustering_app = typer.Typer(
    name="clustering",
    help="Phase 2 — Semantic Clustering Engine",
    no_args_is_help=True,
)


def _get_engine(config_path: str) -> ClusteringEngine:
    cfg = load_clustering_config(config_path)
    return ClusteringEngine(cfg)


@clustering_app.command()
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
        help="Recompute clusters even if existing.",
    ),
) -> None:
    """Generate semantic clusters from relationships."""
    console.print("[bold cyan]Phase 2 — Semantic Cluster Generation[/bold cyan]")
    engine = _get_engine(config)
    result = engine.generate(force=force)

    table = Table(title="Cluster Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Clusters", str(result.get("total_clusters", 0)))
    if "total_members" in result:
        table.add_row("Total Members", str(result["total_members"]))
        table.add_row("Total Relationships", str(result.get("total_relationships", 0)))
        table.add_row("Average Cluster Size", str(result.get("average_cluster_size", 0)))
        table.add_row("Average Density", f"{result.get('average_density', 0):.6f}")
        table.add_row("Average Quality", f"{result.get('average_quality', 0):.6f}")
        table.add_row("Low Quality Clusters", str(result.get("low_quality_clusters", 0)))
        table.add_row("Orphan Concepts", str(result.get("orphan_concepts", 0)))
        table.add_row("Singletons", str(result.get("singletons", 0)))
        table.add_row("Provider", str(result.get("provider", "")))
        table.add_row("Algorithm", str(result.get("algorithm", "")))
        table.add_row("Valid", str(result.get("valid", True)))
    if "elapsed_seconds" in result:
        table.add_row("Elapsed Time", f"{result['elapsed_seconds']}s")
    if "reason" in result:
        table.add_row("Status", result["reason"])
    console.print(table)


@clustering_app.command()
def stats(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Display cluster statistics."""
    engine = _get_engine(config)
    s = engine.stats()

    table = Table(title="Cluster Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Clusters", str(s.get("total_clusters", 0)))
    table.add_row("Total Members", str(s.get("total_members", 0)))
    table.add_row("Total Relationships", str(s.get("total_relationships", 0)))
    table.add_row("Average Cluster Size", f"{s.get('average_cluster_size', 0):.2f}")
    table.add_row("Average Density", f"{s.get('average_density', 0):.6f}")
    table.add_row("Average Quality", f"{s.get('average_quality', 0):.6f}")
    table.add_row("Min Cluster Size", str(s.get("cluster_size_min", 0)))
    table.add_row("Max Cluster Size", str(s.get("cluster_size_max", 0)))
    table.add_row("Median Cluster Size", f"{s.get('cluster_size_median', 0):.1f}")
    console.print(table)

    if s.get("cluster_type_counts"):
        ttable = Table(title="By Cluster Type")
        ttable.add_column("Type", style="cyan")
        ttable.add_column("Count", style="green")
        for ctype, count in sorted(s["cluster_type_counts"].items()):
            ttable.add_row(ctype, str(count))
        console.print(ttable)

    if s.get("quality_distribution"):
        qtable = Table(title="Quality Distribution")
        qtable.add_column("Bucket", style="cyan")
        qtable.add_column("Count", style="green")
        for bucket, count in sorted(s["quality_distribution"].items()):
            qtable.add_row(bucket, str(count))
        console.print(qtable)


@clustering_app.command()
def search(
    query_id: str = typer.Argument(..., help="Cluster ID, member ID, or representative ID."),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Search clusters by ID (cluster, member, or representative)."""
    engine = _get_engine(config)
    results = engine.search_clusters(query_id)

    if not results:
        console.print("[yellow]No clusters found for this ID.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Search Results for: {query_id[:40]}")
    table.add_column("#", style="dim")
    table.add_column("Cluster ID", style="cyan")
    table.add_column("Representative", style="yellow")
    table.add_column("Size", style="green")
    table.add_column("Quality", style="green")
    table.add_column("Type", style="green")

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.cluster_id[:16] + "...",
            r.representative_id[:16] + "...",
            str(r.member_count),
            f"{r.quality_score:.4f}",
            r.cluster_type.value,
        )

    console.print(table)


@clustering_app.command()
def verify(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Verify integrity of stored clusters."""
    engine = _get_engine(config)
    result = engine.verify()

    table = Table(title="Cluster Verification")
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="green")
    table.add_row("Valid", str(result["valid"]))
    table.add_row("Clusters Checked", str(result["clusters_checked"]))
    table.add_row("Members Checked", str(result["members_checked"]))
    table.add_row("Issues Found", str(len(result["issues"])))
    console.print(table)

    if result["issues"]:
        itable = Table(title="Issues")
        itable.add_column("Severity", style="cyan")
        itable.add_column("Code", style="yellow")
        itable.add_column("Message", style="green")
        itable.add_column("Cluster ID", style="dim")
        for issue in result["issues"][:20]:
            itable.add_row(
                issue["severity"],
                issue["code"],
                issue["message"][:60],
                (issue.get("cluster_id") or "")[:16],
            )
        console.print(itable)


if __name__ == "__main__":
    clustering_app()
