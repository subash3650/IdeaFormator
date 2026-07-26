"""CLI commands for the Knowledge Graph Reasoning Engine."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from phase2.reasoning.config import load_reasoning_config
from phase2.reasoning.engine import ReasoningEngine
from phase2.reasoning.exporter import ReasoningExporter

console = Console()
reasoning_app = typer.Typer(
    name="reasoning",
    help="Phase 3, Module 2 — Knowledge Graph Reasoning Engine",
    no_args_is_help=True,
)


def _get_engine(config_path: str) -> ReasoningEngine:
    cfg = load_reasoning_config(config_path)
    return ReasoningEngine(cfg)


@reasoning_app.command()
def reason(
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
        help="Force reasoning even if cache is valid.",
    ),
) -> None:
    """Run the full reasoning pipeline on the knowledge graph."""
    engine = _get_engine(config)
    result = engine.reason(force=force)

    table = Table(title="Knowledge Graph Reasoning — Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Run ID", result.get("run_id", "")[:16])
    table.add_row("Inferences", str(result.get("inference_count", 0)))
    table.add_row("Reasoning Chains", str(result.get("chain_count", 0)))
    table.add_row("Root Causes", str(result.get("root_cause_count", 0)))
    table.add_row("Evidence Aggregations", str(result.get("evidence_aggregation_count", 0)))
    table.add_row("Explanations", str(result.get("explanation_count", 0)))
    table.add_row("Elapsed Time", f"{result.get('elapsed_seconds', 0):.2f}s")
    table.add_row("Cache Hit", str(result.get("cache_hit", False)))
    rules = result.get("rules_applied", [])
    if rules:
        table.add_row("Rules Applied", ", ".join(rules))
    console.print(table)

    firing = result.get("rule_firing_counts", {})
    if firing:
        ftable = Table(title="Rule Firing Counts")
        ftable.add_column("Rule", style="cyan")
        ftable.add_column("Firings", style="green")
        for rule_name, count in sorted(firing.items()):
            ftable.add_row(rule_name, str(count))
        console.print(ftable)


@reasoning_app.command()
def explain(
    inference_id: str = typer.Argument(
        ..., help="Inference ID to explain."
    ),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    format: str = typer.Option(
        "template",
        "--format",
        "-f",
        help="Explanation format: template, structured, markdown.",
    ),
) -> None:
    """Explain a specific inference."""
    engine = _get_engine(config)
    store = engine.store
    chains = store.load_chains()
    inferences = store.load_inferences()
    explanations = _load_explanations(store)

    matching = [e for e in explanations if e.get("inference_id") == inference_id]
    if matching:
        for exp in matching:
            console.print(f"[bold cyan]{exp.get('title', 'Explanation')}[/bold cyan]")
            console.print(exp.get("raw_text", exp.get("summary", "")))
        return

    inf = next((i for i in inferences if i.inference_id == inference_id), None)
    if inf:
        console.print("[cyan]Inference:[/cyan]")
        console.print(f"  Type: {inf.inference_type.value}")
        console.print(f"  Confidence: {inf.confidence:.4f}")
        console.print(f"  Chain ID: {inf.chain_id[:24]}...")
        console.print(f"  Provenance: {', '.join(inf.provenance[:5])}")
        if len(inf.provenance) > 5:
            console.print(f"  ... and {len(inf.provenance) - 5} more nodes")
        chain = next((c for c in chains if c.inference_id == inference_id), None)
        if chain:
            console.print(f"\n[cyan]Reasoning Steps: {len(chain.steps)}[/cyan]")
            for step in chain.steps:
                console.print(f"  {step.step_id + 1}. [{step.rule_name}] confidence: {step.confidence_delta:.2f}")
        return

    console.print(f"[yellow]No explanation found for inference: {inference_id}[/yellow]")


@reasoning_app.command(name="root-cause")
def root_cause(
    effect_id: str = typer.Option(
        None,
        "--effect-id",
        "-e",
        help="Filter root causes by effect node ID.",
    ),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of root causes to display.",
    ),
) -> None:
    """Display discovered root causes."""
    engine = _get_engine(config)
    store = engine.store
    root_causes = store.load_root_causes()
    if effect_id:
        root_causes = [rc for rc in root_causes if rc.effect_node_id == effect_id]
    root_causes = root_causes[:limit]

    if not root_causes:
        console.print("[yellow]No root causes found. Run 'reason' first.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Root Causes ({len(root_causes)} found)")
    table.add_column("#", style="dim")
    table.add_column("Cause", style="cyan")
    table.add_column("Effect", style="yellow")
    table.add_column("Hops", style="green")
    table.add_column("Confidence", style="green")
    table.add_column("Impact", style="green")
    table.add_column("Score", style="green")

    for i, rc in enumerate(root_causes, 1):
        table.add_row(
            str(i),
            rc.cause_label[:30],
            rc.effect_label[:30],
            str(rc.path_length),
            f"{rc.propagated_confidence:.2f}",
            str(rc.transitive_impact_count),
            f"{rc.ranking_score:.4f}",
        )
    console.print(table)


@reasoning_app.command()
def chains(
    inference_id: str = typer.Option(
        None,
        "--inference-id",
        "-i",
        help="Filter chains by inference ID.",
    ),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of chains to display.",
    ),
) -> None:
    """Display reasoning chains."""
    engine = _get_engine(config)
    store = engine.store
    chains = store.load_chains()
    if inference_id:
        chains = [c for c in chains if c.inference_id == inference_id]
    chains = chains[:limit]

    if not chains:
        console.print("[yellow]No reasoning chains found. Run 'reason' first.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Reasoning Chains ({len(chains)} found)")
    table.add_column("#", style="dim")
    table.add_column("Chain ID", style="cyan")
    table.add_column("Inference ID", style="yellow")
    table.add_column("Steps", style="green")
    table.add_column("Confidence", style="green")

    for i, c in enumerate(chains, 1):
        table.add_row(
            str(i),
            c.chain_id[:16] + "...",
            c.inference_id[:16] + "...",
            str(len(c.steps)),
            f"{c.total_confidence:.2f}",
        )
    console.print(table)


@reasoning_app.command()
def stats(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON.",
    ),
) -> None:
    """Display reasoning statistics."""
    engine = _get_engine(config)
    s = engine.stats()

    if json_output:
        console.print(json.dumps(s, indent=2, default=str))
        return

    table = Table(title="Reasoning Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Graph Nodes", str(s.get("graph_nodes", 0)))
    table.add_row("Graph Edges", str(s.get("graph_edges", 0)))
    table.add_row("Stored Inferences", str(s.get("inferences", 0)))
    table.add_row("Stored Chains", str(s.get("chains", 0)))
    table.add_row("Stored Root Causes", str(s.get("root_causes", 0)))
    table.add_row("Cache Valid", str(s.get("cache_valid", False)))
    table.add_row("Run ID", s.get("run_id", "")[:20])
    console.print(table)


@reasoning_app.command()
def cache(
    action: str = typer.Argument(
        ..., help="Cache action: clear."
    ),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Manage reasoning cache."""
    if action != "clear":
        console.print(f"[red]Unknown action: {action}. Use 'clear'.[/red]")
        raise typer.Exit(1)
    engine = _get_engine(config)
    engine.clear_cache()
    console.print("[green]Reasoning cache cleared.[/green]")


@reasoning_app.command()
def export(
    format: str = typer.Option(
        "all",
        "--format",
        "-f",
        help="Export format: all, report, statistics, dashboard, summary.",
    ),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Export reasoning results."""
    engine = _get_engine(config)
    exporter = ReasoningExporter(engine.store)

    paths: dict[str, str] = {}
    if format in ("all", "report"):
        paths["report"] = str(exporter.export_report())
    if format in ("all", "statistics"):
        paths["statistics"] = str(exporter.export_statistics())
    if format in ("all", "dashboard"):
        paths["dashboard"] = str(exporter.export_dashboard())
    if format in ("all", "summary"):
        paths["summary"] = str(exporter.export_summary())

    table = Table(title="Export Results")
    table.add_column("Format", style="cyan")
    table.add_column("Path", style="green")
    for fmt, p in sorted(paths.items()):
        table.add_row(fmt, p)
    console.print(table)


def _load_explanations(store) -> list[dict]:
    path = store.explanations_path
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []
