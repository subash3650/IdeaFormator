"""CLI commands for the Knowledge Graph Infrastructure."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from phase2.knowledge_graph.config import load_knowledge_graph_config
from phase2.knowledge_graph.engine import KnowledgeGraphEngine

console = Console()
kg_app = typer.Typer(
    name="knowledge-graph",
    help="Phase 3, Module 1 — Knowledge Graph Infrastructure",
    no_args_is_help=True,
)


def _get_engine(config_path: str) -> KnowledgeGraphEngine:
    cfg = load_knowledge_graph_config(config_path)
    return KnowledgeGraphEngine(cfg)


@kg_app.command()
def build(
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
        help="Force build even if assets are missing.",
    ),
) -> None:
    """Build the knowledge graph from pipeline assets."""
    console.print("[bold cyan]Phase 3, Module 1 — Knowledge Graph Build[/bold cyan]")
    engine = _get_engine(config)
    result = engine.build(force=force)

    table = Table(title="Knowledge Graph Build Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Nodes", str(result.get("node_count", 0)))
    table.add_row("Edges", str(result.get("edge_count", 0)))
    table.add_row("Density", f"{result.get('density', 0):.6f}")
    table.add_row("Connected Components", str(result.get("connected_components", 0)))
    table.add_row("Largest Component Size", str(result.get("largest_component_size", 0)))
    table.add_row("Health Score", f"{result.get('health_score', 0):.1f}/100")
    table.add_row("Health Status", result.get("health_status", ""))
    table.add_row("Valid", str(result.get("valid", False)))
    if "elapsed_seconds" in result:
        table.add_row("Elapsed Time", f"{result['elapsed_seconds']:.2f}s")
    console.print(table)


@kg_app.command()
def search(
    query: str = typer.Argument(..., help="Node ID or label query."),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    type: str = typer.Option(
        "",
        "--type",
        "-t",
        help="Filter by node type (e.g., observation, evidence).",
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        "-k",
        help="Maximum number of results.",
    ),
) -> None:
    """Search the knowledge graph by node ID or label."""
    engine = _get_engine(config)
    results = engine.search(query, top_k=top_k, type_filter=type)

    if not results:
        console.print(f"[yellow]No results found for: {query}[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Search Results: {query[:40]}")
    table.add_column("#", style="dim")
    table.add_column("Node ID", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Label", style="green")
    table.add_column("Confidence", style="green")

    for i, r in enumerate(results[:top_k], 1):
        table.add_row(
            str(i),
            r.get("node_id", "")[:16] + "...",
            r.get("node_type", ""),
            r.get("label", "")[:40],
            f"{r.get('confidence', 0):.4f}",
        )
    console.print(table)


@kg_app.command()
def stats(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Display knowledge graph statistics."""
    engine = _get_engine(config)
    s = engine.stats()

    table = Table(title="Knowledge Graph Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Nodes", str(s.get("node_count", 0)))
    table.add_row("Edges", str(s.get("edge_count", 0)))
    table.add_row("Density", f"{s.get('density', 0):.6f}")
    table.add_row("Connected Components", str(s.get("connected_components", 0)))
    table.add_row("Average Confidence", f"{s.get('avg_confidence', 0):.4f}")
    table.add_row("Average Degree", f"{s.get('avg_degree', 0):.4f}")
    table.add_row("Orphan Nodes", str(s.get("orphan_node_count", 0)))
    table.add_row("Orphan Edges", str(s.get("orphan_edge_count", 0)))
    table.add_row("Health Score", f"{s.get('health_score', 0):.1f}/100")
    console.print(table)

    if s.get("type_distribution"):
        ttable = Table(title="Node Type Distribution")
        ttable.add_column("Type", style="cyan")
        ttable.add_column("Count", style="green")
        for ntype, count in sorted(s["type_distribution"].items()):
            ttable.add_row(ntype, str(count))
        console.print(ttable)

    if s.get("edge_type_distribution"):
        etable = Table(title="Edge Type Distribution")
        etable.add_column("Type", style="cyan")
        etable.add_column("Count", style="green")
        for etype, count in sorted(s["edge_type_distribution"].items()):
            etable.add_row(etype, str(count))
        console.print(etable)


@kg_app.command()
def verify(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Verify knowledge graph integrity."""
    engine = _get_engine(config)
    result = engine.verify()

    table = Table(title="Knowledge Graph Verification")
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="green")
    table.add_row("Valid", str(result.get("valid", False)))
    table.add_row("Nodes Checked", str(result.get("node_count", 0)))
    table.add_row("Edges Checked", str(result.get("edge_count", 0)))
    table.add_row("Errors", str(len(result.get("errors", []))))
    table.add_row("Warnings", str(len(result.get("warnings", []))))
    console.print(table)

    if result.get("errors"):
        etable = Table(title="Errors")
        etable.add_column("Message", style="red")
        for err in result["errors"][:10]:
            etable.add_row(err[:80])
        console.print(etable)

    if result.get("warnings"):
        wtable = Table(title="Warnings")
        wtable.add_column("Message", style="yellow")
        for warn in result["warnings"][:10]:
            wtable.add_row(warn[:80])
        console.print(wtable)


@kg_app.command()
def export(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    format: str = typer.Option(
        "all",
        "--format",
        "-f",
        help="Export format: all, gexf, json, csv.",
    ),
) -> None:
    """Export knowledge graph in various formats."""
    engine = _get_engine(config)
    result = engine.export(format=format)

    table = Table(title="Export Results")
    table.add_column("Format", style="cyan")
    table.add_column("Path", style="green")

    for fmt, path in sorted(result.items()):
        if isinstance(path, dict):
            for subfmt, subpath in path.items():
                table.add_row(f"{fmt}/{subfmt}", str(subpath))
        else:
            table.add_row(fmt, str(path))
    console.print(table)


if __name__ == "__main__":
    kg_app()
