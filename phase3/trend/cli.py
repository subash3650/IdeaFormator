"""Typer CLI for the Trend Intelligence Engine."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from phase3.trend.config import TrendConfig, load_trend_config
from phase3.trend.engine import TrendEngine
from phase3.trend.store import TrendStore

trend_app = typer.Typer(name="trend", help="Trend Intelligence Engine")
console = Console()


@trend_app.command()
def snapshot(
    knowledge_dir: str = typer.Option(
        "pain_intelligence/knowledge",
        "--knowledge-dir", "-k",
        help="Path to knowledge base directory.",
    ),
    run_id: str = typer.Option(
        None, "--run-id", "-r",
        help="Run ID for the snapshot. Auto-generated if not provided.",
    ),
) -> None:
    """Create a snapshot of current pipeline assets."""
    console.print("[bold cyan]Trend Snapshot[/bold cyan]")

    cfg = TrendConfig(output_dir=Path(knowledge_dir) / "assets" / "phase3")
    engine = TrendEngine(cfg, run_id=run_id)
    result = engine.create_snapshot(knowledge_dir)

    table = Table(title="Snapshot Created")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Snapshot ID", result["snapshot_id"])
    table.add_row("Run ID", result["run_id"])
    table.add_row("Timestamp", result["timestamp"])
    table.add_row("Observations", str(result["observation_count"]))
    table.add_row("Evidence", str(result["evidence_count"]))
    table.add_row("Signals", str(result["signal_count"]))
    table.add_row("Opportunities", str(result["opportunity_count"]))
    console.print(table)


@trend_app.command()
def generate(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to YAML configuration file.",
    ),
    knowledge_dir: str = typer.Option(
        "pain_intelligence/knowledge",
        "--knowledge-dir", "-k",
        help="Path to knowledge base directory.",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force regeneration, bypassing cache.",
    ),
) -> None:
    """Run trend detection on historical snapshots."""
    console.print("[bold cyan]Trend Intelligence Engine[/bold cyan] — Generate")

    cfg = load_trend_config(config)
    if not cfg or cfg.output_dir == Path("pain_intelligence/knowledge/assets/phase3"):
        cfg = TrendConfig(
            output_dir=Path(knowledge_dir) / "assets" / "phase3",
            knowledge_dir=Path(knowledge_dir),
        )

    engine = TrendEngine(cfg)
    result = engine.generate(knowledge_dir, force=force)

    table = Table(title="Trend Detection Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Trends", str(result["total_trends"]))
    table.add_row("Growing", str(result["growing"]))
    table.add_row("Declining", str(result["declining"]))
    table.add_row("Emerging", str(result["emerging"]))
    table.add_row("Stable", str(result["stable"]))
    table.add_row("Average Score", str(result["avg_trend_score"]))
    table.add_row("Cache Hit", str(result["cache_hit"]))
    table.add_row("Elapsed", f"{result['elapsed_seconds']}s")
    console.print(table)


@trend_app.command()
def top(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to YAML configuration file.",
    ),
    knowledge_dir: str = typer.Option(
        "pain_intelligence/knowledge",
        "--knowledge-dir", "-k",
        help="Path to knowledge base directory.",
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of top trends."),
    trend_type: str = typer.Option(
        None, "--type", help="Filter by trend type (growing, declining, emerging).",
    ),
) -> None:
    """Show top trends sorted by score."""
    cfg = TrendConfig(
        output_dir=Path(knowledge_dir) / "assets" / "phase3",
        knowledge_dir=Path(knowledge_dir),
        top_k=limit,
    )
    engine = TrendEngine(cfg)
    trends = engine.store.load_trends()

    if trend_type:
        trends = [t for t in trends if t.trend_type.value == trend_type]

    trends = sorted(trends, key=lambda t: -t.metrics.trend_score)[:limit]

    table = Table(title=f"Top {limit} Trends{' (' + trend_type + ')' if trend_type else ''}")
    table.add_column("Rank", style="dim")
    table.add_column("ID", style="dim")
    table.add_column("Title", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Direction", style="magenta")
    table.add_column("Growth %", style="green")
    table.add_column("Score", style="blue")

    for i, t in enumerate(trends, 1):
        table.add_row(
            str(i),
            t.trend_id[:8],
            t.title[:40],
            t.trend_type.value,
            t.trend_direction.value,
            f"{t.metrics.growth_pct:.1f}%",
            f"{t.metrics.trend_score:.3f}",
        )
    console.print(table)


@trend_app.command()
def show(
    trend_id: str = typer.Argument(..., help="Trend ID to display"),
    knowledge_dir: str = typer.Option(
        "pain_intelligence/knowledge",
        "--knowledge-dir", "-k",
        help="Path to knowledge base directory.",
    ),
) -> None:
    """Show detailed information about a specific trend."""
    from phase3.trend.search import TrendSearch

    cfg = TrendConfig(output_dir=Path(knowledge_dir) / "assets" / "phase3")
    engine = TrendEngine(cfg)
    trends = engine.store.load_trends()
    searcher = TrendSearch(trends)
    t = searcher.find_by_id(trend_id)

    if t is None:
        console.print(f"[red]Trend not found: {trend_id}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]{t.title}[/bold cyan]")
    console.print(f"  ID:        {t.trend_id}")
    console.print(f"  Type:      {t.trend_type.value}")
    console.print(f"  Direction: {t.trend_direction.value}")
    console.print(f"  Subject:   {t.trend_subject.value} ({t.subject_label})")
    console.print(f"  Summary:   {t.summary}")
    console.print(f"")
    console.print("[bold]Metrics:[/bold]")
    console.print(f"  Growth %:      {t.metrics.growth_pct:.2f}")
    console.print(f"  Velocity:      {t.metrics.velocity:.4f}")
    console.print(f"  Momentum:      {t.metrics.momentum:.4f}")
    console.print(f"  Confidence:    {t.metrics.confidence:.4f}")
    console.print(f"  Trend Score:   {t.metrics.trend_score:.4f}")
    console.print(f"  Duration:      {t.metrics.duration_days} days")
    console.print(f"  Snapshots:     {t.metrics.snapshot_count}")
    console.print(f"  Observations:  {t.metrics.total_observations}")

    if t.affected_platforms:
        console.print(f"\n[bold]Platforms:[/bold] {', '.join(t.affected_platforms)}")
    if t.affected_products:
        console.print(f"[bold]Products:[/bold] {', '.join(t.affected_products)}")
    if t.affected_companies:
        console.print(f"[bold]Companies:[/bold] {', '.join(t.affected_companies)}")
    if t.affected_technologies:
        console.print(f"[bold]Technologies:[/bold] {', '.join(t.affected_technologies)}")
    if t.correlations:
        console.print(f"\n[bold]Correlations ({len(t.correlations)}):[/bold]")
        for c in t.correlations:
            console.print(f"  {c.correlation_type.value}: {c.related_entity_id} (strength={c.correlation_strength})")


@trend_app.command()
def stats(
    knowledge_dir: str = typer.Option(
        "pain_intelligence/knowledge",
        "--knowledge-dir", "-k",
        help="Path to knowledge base directory.",
    ),
) -> None:
    """Show trend statistics."""
    cfg = TrendConfig(output_dir=Path(knowledge_dir) / "assets" / "phase3")
    engine = TrendEngine(cfg)
    s = engine.stats()

    table = Table(title="Trend Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in s.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                table.add_row(f"{k}.{sk}", str(sv))
        else:
            table.add_row(k, str(v))
    console.print(table)


@trend_app.command()
def export(
    knowledge_dir: str = typer.Option(
        "pain_intelligence/knowledge",
        "--knowledge-dir", "-k",
        help="Path to knowledge base directory.",
    ),
) -> None:
    """Export trend reports and dashboards."""
    console.print("[bold cyan]Exporting Trend Reports[/bold cyan]")

    cfg = TrendConfig(output_dir=Path(knowledge_dir) / "assets" / "phase3")
    store = TrendStore(cfg.trend_dir)

    from phase3.trend.exporter import TrendExporter
    exporter = TrendExporter(store)
    paths = exporter.export_all()

    table = Table(title="Exported Files")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="green")
    for name, path in paths.items():
        table.add_row(name, str(path))
    console.print(table)
    console.print(f"[green]Reports written to {cfg.trend_dir}/[/green]")
