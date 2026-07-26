"""CLI commands for the Opportunity Discovery Engine."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from phase3.opportunity.config import load_opportunity_config
from phase3.opportunity.engine import OpportunityEngine
from phase3.opportunity.exporter import OpportunityExporter

console = Console()
opportunity_app = typer.Typer(
    name="opportunity",
    help="Phase 3, Module 3 — Opportunity Discovery Engine",
    no_args_is_help=True,
)


def _get_engine(config_path: str) -> OpportunityEngine:
    cfg = load_opportunity_config(config_path)
    return OpportunityEngine(cfg)


@opportunity_app.command()
def discover(
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
        help="Force discovery even if cache is valid.",
    ),
) -> None:
    """Run the full opportunity discovery pipeline."""
    engine = _get_engine(config)
    result = engine.discover(force=force)

    table = Table(title="Opportunity Discovery — Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Run ID", result.get("run_id", "")[:16])
    table.add_row("Total Candidates", str(result.get("total_candidates", 0)))
    table.add_row("Total Opportunities", str(result.get("total_opportunities", 0)))
    table.add_row("Scored", str(result.get("scored_opportunities", 0)))
    table.add_row("Ranked", str(result.get("ranked_opportunities", 0)))
    table.add_row("Average Score", f"{result.get('avg_opportunity_score', 0):.4f}")
    table.add_row("Top Score", f"{result.get('top_score', 0):.4f}")
    table.add_row("Elapsed", f"{result.get('elapsed_seconds', 0):.2f}s")
    table.add_row("Cache Hit", str(result.get("cache_hit", False)))
    scoring_providers = result.get("scoring_providers", [])
    if scoring_providers:
        table.add_row("Scoring Providers", ", ".join(scoring_providers))
    bm_providers = result.get("business_model_providers", [])
    if bm_providers:
        table.add_row("Business Model Providers", ", ".join(bm_providers))
    console.print(table)

    rec_dist = result.get("recommendation_distribution", {})
    if rec_dist:
        rt = Table(title="Recommendation Distribution")
        rt.add_column("Type", style="cyan")
        rt.add_column("Count", style="green")
        for k, v in sorted(rec_dist.items()):
            rt.add_row(k, str(v))
        console.print(rt)

    bm_dist = result.get("business_model_distribution", {})
    if bm_dist:
        bt = Table(title="Business Model Distribution")
        bt.add_column("Model", style="cyan")
        bt.add_column("Count", style="green")
        for k, v in sorted(bm_dist.items()):
            bt.add_row(k, str(v))
        console.print(bt)


@opportunity_app.command()
def top(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Number of top opportunities to display.",
    ),
) -> None:
    """Display top-ranked opportunities."""
    engine = _get_engine(config)
    opportunities = engine.store.load_opportunities()
    if not opportunities:
        console.print("[yellow]No opportunities found. Run 'discover' first.[/yellow]")
        return

    table = Table(title=f"Top {min(limit, len(opportunities))} Opportunities")
    table.add_column("Rank", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Score", style="green")
    table.add_column("Recommendation", style="yellow")
    table.add_column("Business Model", style="blue")
    table.add_column("Confidence", style="magenta")

    for opp in sorted(opportunities, key=lambda o: -o.opportunity_score)[:limit]:
        table.add_row(
            str(opp.rank),
            opp.title[:60],
            f"{opp.opportunity_score:.4f}",
            opp.recommendation_type.value,
            opp.suggested_business_model.value,
            f"{opp.confidence.final_confidence:.2f}",
        )
    console.print(table)


@opportunity_app.command()
def show(
    opportunity_id: str = typer.Argument(
        ..., help="Opportunity ID to display."
    ),
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Show detailed information about a specific opportunity."""
    engine = _get_engine(config)
    opportunities = engine.store.load_opportunities()
    opp = next((o for o in opportunities if o.opportunity_id == opportunity_id), None)
    if not opp:
        console.print(f"[yellow]Opportunity not found: {opportunity_id}[/yellow]")
        return

    console.print(f"[bold cyan]Opportunity: {opp.title}[/bold cyan]")
    console.print(f"  ID: {opp.opportunity_id}")
    console.print(f"  Rank: {opp.rank}")
    console.print(f"  Score: {opp.opportunity_score:.4f}")
    console.print(f"  Recommendation: {opp.recommendation_type.value}")
    console.print(f"  Business Model: {opp.suggested_business_model.value}")
    console.print(f"")
    console.print(f"  [cyan]Summary:[/cyan] {opp.summary}")
    console.print(f"  [cyan]Solution:[/cyan] {opp.suggested_solution}")
    console.print(f"")
    console.print(f"  [cyan]Confidence:[/cyan]")
    console.print(f"    Reasoning: {opp.confidence.reasoning_confidence:.2f}")
    console.print(f"    Evidence:  {opp.confidence.evidence_confidence:.2f}")
    console.print(f"    Graph:     {opp.confidence.graph_confidence:.2f}")
    console.print(f"    Market:    {opp.confidence.market_confidence:.2f}")
    console.print(f"    Final:     {opp.confidence.final_confidence:.2f}")
    console.print(f"")
    console.print(f"  [cyan]Scoring Breakdown:[/cyan]")
    sb = opp.scoring_breakdown
    console.print(f"    Pain Severity:     {sb.pain_severity:.2f}")
    console.print(f"    Frequency:         {sb.frequency:.2f}")
    console.print(f"    Trend:             {sb.trend:.2f}")
    console.print(f"    Evidence Count:    {sb.evidence_count:.2f}")
    console.print(f"    Reasoning Conf:    {sb.reasoning_confidence:.2f}")
    console.print(f"    Cluster Density:   {sb.cluster_density:.2f}")
    console.print(f"    Cross Platform:    {sb.cross_platform:.2f}")
    console.print(f"    Market Coverage:   {sb.market_coverage:.2f}")
    console.print(f"    Competition:       {sb.competition:.2f}")
    console.print(f"    Feasibility:       {sb.feasibility:.2f}")
    console.print(f"    Novelty:           {sb.novelty:.2f}")
    console.print(f"")
    if opp.affected_products:
        console.print(f"  [cyan]Affected Products:[/cyan] {', '.join(opp.affected_products[:5])}")
    if opp.affected_companies:
        console.print(f"  [cyan]Affected Companies:[/cyan] {', '.join(opp.affected_companies[:5])}")
    if opp.supporting_evidence:
        console.print(f"  [cyan]Evidence Count:[/cyan] {len(opp.supporting_evidence)}")
    if opp.reasoning_chain_ids:
        console.print(f"  [cyan]Reasoning Chains:[/cyan] {len(opp.reasoning_chain_ids)}")
    console.print(f"  [cyan]Market Size:[/cyan] {opp.estimated_market_size.value}")
    console.print(f"  [cyan]Status:[/cyan] {opp.status.value}")


@opportunity_app.command()
def stats(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Show discovery statistics."""
    engine = _get_engine(config)
    stats_data = engine.stats()

    table = Table(title="Opportunity Engine — Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Opportunities", str(stats_data.get("total_opportunities", 0)))
    table.add_row("Average Score", f"{stats_data.get('avg_score', 0):.4f}")
    table.add_row("Min Score", f"{stats_data.get('min_score', 0):.4f}")
    table.add_row("Max Score", f"{stats_data.get('max_score', 0):.4f}")
    table.add_row("Run ID", stats_data.get("run_id", "")[:16])
    table.add_row("Cache Hit", str(stats_data.get("cache_hit", False)))
    console.print(table)

    rec_dist = stats_data.get("recommendation_distribution", {})
    if rec_dist:
        rt = Table(title="Recommendation Distribution")
        rt.add_column("Type", style="cyan")
        rt.add_column("Count", style="green")
        for k, v in sorted(rec_dist.items()):
            rt.add_row(k, str(v))
        console.print(rt)

    bm_dist = stats_data.get("business_model_distribution", {})
    if bm_dist:
        bt = Table(title="Business Model Distribution")
        bt.add_column("Model", style="cyan")
        bt.add_column("Count", style="green")
        for k, v in sorted(bm_dist.items()):
            bt.add_row(k, str(v))
        console.print(bt)


@opportunity_app.command()
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
        help="Export format: report, statistics, dashboard, summary, csv, all.",
    ),
) -> None:
    """Export opportunity reports."""
    engine = _get_engine(config)
    exporter = OpportunityExporter(engine.store)
    results: dict[str, str] = {}

    if format in ("all", "report"):
        results["report"] = str(exporter.export_report())
    if format in ("all", "statistics"):
        results["statistics"] = str(exporter.export_statistics())
    if format in ("all", "dashboard"):
        results["dashboard_json"] = str(exporter.export_dashboard())
        results["dashboard_txt"] = str(exporter.export_dashboard_text())
    if format in ("all", "summary"):
        results["summary"] = str(exporter.export_summary())
    if format in ("all", "csv"):
        results["csv"] = str(exporter.export_csv())

    table = Table(title="Exported Files")
    table.add_column("Format", style="cyan")
    table.add_column("Path", style="green")
    for fmt, path in results.items():
        table.add_row(fmt, path)
    console.print(table)
