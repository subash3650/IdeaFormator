from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from phase2.evaluation.exporter import export_all
from phase2.evaluation.evaluator import EvaluationOrchestrator

evaluate_app = typer.Typer(
    name="evaluate",
    help="Evaluate pipeline quality across all stages.",
    add_completion=False,
)
console = Console()


@evaluate_app.command()
def evaluate(
    knowledge_dir: str = typer.Option(
        "pain_intelligence/knowledge",
        "--knowledge-dir",
        "-k",
        help="Path to knowledge base directory.",
    ),
    output_dir: str = typer.Option(
        "evaluation_reports",
        "--output",
        "-o",
        help="Output directory for evaluation reports.",
    ),
    dashboard: bool = typer.Option(
        False,
        "--dashboard",
        "-d",
        help="Generate dashboard files in addition to reports.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output summary as JSON to stdout.",
    ),
) -> None:
    """Run full pipeline quality evaluation."""
    console.print("[bold cyan]Pain Intelligence[/bold cyan] — Pipeline Evaluation")

    orchestrator = EvaluationOrchestrator(knowledge_dir=knowledge_dir)
    result = orchestrator.evaluate()

    # Print health scores
    table = Table(title="Pipeline Health Scores")
    table.add_column("Stage", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Status")

    stages = [
        ("Documents", result.documents.health),
        ("Observations", result.observations.health),
        ("Evidence", result.evidence.health),
        ("Signals", result.signals.health),
        ("Embeddings", result.embeddings.health),
        ("Relationships", result.relationships.health),
        ("Clusters", result.clusters.health),
    ]
    for name, health in stages:
        status = "✅" if health.score >= 70 else "⚠️" if health.score >= 40 else "❌"
        table.add_row(name, f"{health.score:.1f}/100", status)
    table.add_section()
    table.add_row("[bold]Overall", f"[bold]{result.overall_health_score:.1f}/100", "")
    console.print(table)

    if result.worst_stage:
        console.print(f"\n[yellow]Worst stage: {result.worst_stage}[/yellow]")

    if result.all_warnings:
        wtable = Table(title="Warnings")
        wtable.add_column("Warning")
        for w in result.all_warnings:
            wtable.add_row(f"[yellow]{w}[/yellow]")
        console.print(wtable)

    if result.recommendations:
        rtable = Table(title="Recommendations")
        rtable.add_column("Recommendation")
        for r in result.recommendations:
            rtable.add_row(f"[cyan]{r}[/cyan]")
        console.print(rtable)

    # Export
    export_all(result, output_dir=output_dir)

    if dashboard:
        from phase2.evaluation.dashboard import generate_dashboard
        dash_paths = generate_dashboard(result, output_dir=output_dir)
        console.print(f"[green]Dashboard written to {output_dir}/[/green]")

    console.print(f"[green]Reports written to {output_dir}/[/green]")

    # Optional JSON output
    if json_output:
        import json
        from phase2.evaluation.reports import generate_summary
        console.print(json.dumps(generate_summary(result), indent=2, default=str))
