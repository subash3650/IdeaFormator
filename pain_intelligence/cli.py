"""Typer CLI for the Pain Intelligence Engine."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pain_intelligence.pipeline.orchestrator import Orchestrator
from pain_intelligence.ingestion.cli import ingest_app
from phase2.clustering.cli import clustering_app
from phase2.embeddings.cli import embeddings_app
from phase2.similarity.cli import similarity_app
from phase2.reasoning.cli import reasoning_app
from phase2.knowledge_graph.cli import kg_app
from phase3.opportunity.cli import opportunity_app
from phase3.trend.cli import trend_app
from phase3.presentation.cli import presentation_app
from phase4.copilot.cli import copilot_app

app = typer.Typer(
    name="pain-intelligence",
    help="Pain Intelligence Engine - Data Ingestion & Preprocessing Pipeline",
    add_completion=False,
)
app.add_typer(ingest_app)
app.add_typer(clustering_app)
app.add_typer(embeddings_app)
app.add_typer(similarity_app)
app.add_typer(reasoning_app)
app.add_typer(kg_app)
app.add_typer(opportunity_app)
app.add_typer(trend_app)
app.add_typer(presentation_app)
app.add_typer(copilot_app)
console = Console()


@app.command()
def run(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Run the full data ingestion and preprocessing pipeline."""
    console.print("[bold green]Pain Intelligence Engine[/bold green] v0.1.0")
    console.print(f"Config: {config}")

    if not Path(config).exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)

    orchestrator = Orchestrator(config_path=config)
    stats = orchestrator.run()

    _print_summary(stats)


@app.command()
def analyze(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    data: str = typer.Option(
        None,
        "--data",
        "-d",
        help="Path to processed dataset. Defaults to knowledge/processed/processed.parquet.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode with detailed resolution metadata.",
    ),
    sample: int = typer.Option(
        0,
        "--sample",
        help="Sample N documents (0 = no sampling).",
    ),
) -> None:
    """Run the knowledge extraction pipeline (Phase 1.5)."""
    console.print("[bold cyan]Pain Intelligence[/bold cyan] — Knowledge Extraction Engine v1.5.0")

    if data is None:
        ingestion_path = Path("pain_intelligence/knowledge/processed/processed.parquet")
        legacy_path = Path("outputs/processed.parquet")
        if ingestion_path.exists():
            data = str(ingestion_path)
        elif legacy_path.exists():
            data = str(legacy_path)
            console.print("[yellow]Using legacy outputs/processed.parquet. "
                          "Run 'build-dataset' to create knowledge/processed/.[/yellow]")
        else:
            console.print("[red]No dataset found. Run 'ingest run' then 'build-dataset'.[/red]")
            raise typer.Exit(1)

    console.print(f"Config: {config}, Data: {data}")

    if not Path(data).exists():
        console.print(f"[red]Dataset not found: {data}[/red]")
        raise typer.Exit(1)

    from pain_intelligence.intelligence.engine import IntelligenceEngine

    engine = IntelligenceEngine(config_path=config, debug=debug)
    result = engine.run(data_path=data)

    table = Table(title="Knowledge Extraction Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Status", result.get("status", "unknown"))
    table.add_row("Run ID", result.get("run_id", "unknown"))
    table.add_row("Observations", str(result.get("observations_count", 0)))
    table.add_row("Evidence Records", str(result.get("evidence_count", 0)))
    table.add_row("Problem Signals", str(result.get("signal_count", 0)))
    table.add_row("Input Documents", str(result.get("input_document_count", 0)))
    table.add_row("Elapsed Time", f"{result.get('elapsed_seconds', 0)}s")
    console.print(table)

    # Show adaptive thresholds
    thresholds = result.get("adaptive_thresholds", {})
    if thresholds:
        ttable = Table(title="Adaptive Thresholds")
        ttable.add_column("Threshold", style="cyan")
        ttable.add_column("Value", style="green")
        for k, v in thresholds.items():
            ttable.add_row(k, str(v))
        console.print(ttable)

    # Show filtering stats
    filtering = result.get("filtering", {})
    if filtering:
        ftable = Table(title="Filtering")
        ftable.add_column("Metric", style="cyan")
        ftable.add_column("Count", style="yellow")
        for k, v in filtering.items():
            ftable.add_row(k, str(v))
        console.print(ftable)


@app.command("build-dataset")
def build_dataset(
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path. Defaults to knowledge/processed/processed.parquet.",
    ),
    knowledge_base: str = typer.Option(
        "pain_intelligence/knowledge",
        "--knowledge-base",
        help="Path to the knowledge base directory.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Rebuild even if output already exists.",
    ),
) -> None:
    """Build a unified dataset from all ingestion outputs."""
    console.print("[bold cyan]Build Dataset[/bold cyan]")
    console.print(f"Knowledge base: {knowledge_base}")

    from pain_intelligence.ingestion.dataset_builder import DatasetBuilder

    builder = DatasetBuilder(knowledge_base=knowledge_base)

    result = builder.build(output_path=output, force=force)

    if result.get("status") == "empty":
        console.print("[yellow]No ingestion data found. Run 'ingest run' first.[/yellow]")
        raise typer.Exit(1)

    table = Table(title="Dataset Build Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Documents", str(result.get("total_documents", 0)))
    table.add_row("Duplicates Removed", str(result.get("duplicates_removed", 0)))
    table.add_row("Final Documents", str(result.get("final_documents", 0)))
    table.add_row("Output", result.get("output_path", ""))
    table.add_row("Elapsed Time", f"{result.get('elapsed_seconds', 0)}s")

    console.print(table)

    sources = result.get("sources", {})
    if sources:
        st = Table(title="Per-Source Breakdown")
        st.add_column("Source", style="cyan")
        st.add_column("Documents", style="yellow")
        st.add_column("First Seen", style="dim")
        st.add_column("Last Seen", style="dim")

        for name, info in sources.items():
            st.add_row(
                name,
                str(info.get("documents", 0)),
                str(info.get("first_seen", ""))[:19],
                str(info.get("last_seen", ""))[:19],
            )
        console.print(st)

    console.print("[green]Dataset built successfully.[/green]")
def explore(
    port: int = typer.Option(8501, "--port", "-p", help="Dashboard port."),
) -> None:
    """Launch the Streamlit knowledge explorer (minimal dashboard)."""
    console.print("[bold]Launching Knowledge Explorer...[/bold]")
    console.print(f"Dashboard will be available at http://localhost:{port}")
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / "intelligence" / "dashboard.py"),
        "--server.port", str(port),
        "--server.headless", "true",
    ])


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    platform: str = typer.Option(None, "--platform", help="Filter by platform."),
    country: str = typer.Option(None, "--country", help="Filter by country."),
    limit: int = typer.Option(20, "--limit", "-l", help="Max results."),
) -> None:
    """Search documents in the processed dataset."""
    from pain_intelligence.intelligence.search import DocumentSearch

    engine = DocumentSearch()
    results = engine.search(
        query=query,
        platform=platform,
        country=country,
        limit=limit,
    )

    table = Table(title=f"Search Results: '{query}'")
    table.add_column("ID", style="dim")
    table.add_column("Platform", style="cyan")
    table.add_column("Rating", style="yellow")
    table.add_column("Text Preview", style="green")

    for row in results.iter_rows(named=True):
        doc_id = (row.get("id") or "")[:8]
        plat = row.get("platform", "")
        rating = str(row.get("rating", ""))
        text = (row.get("text") or "")[:100]
        table.add_row(doc_id, plat, rating, text)

    console.print(table)
    console.print(f"[dim]Found {len(results)} results[/dim]")


@app.command()
def validate(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
) -> None:
    """Validate configuration and check dataset files exist."""
    from pain_intelligence.utils.config import load_config, get_nested

    console.print("[bold]Validating configuration...[/bold]")

    try:
        cfg = load_config(config)
        console.print(f"[green]Config loaded: {config}[/green]")
    except Exception as e:
        console.print(f"[red]Config error: {e}[/red]")
        raise typer.Exit(1)

    raw_dir = Path(get_nested(cfg, "paths", "raw_datasets_dir", default="Datasets"))
    if raw_dir.exists():
        csv_files = list(raw_dir.glob("*.csv"))
        console.print(f"Found {len(csv_files)} CSV files in {raw_dir}")
        for f in csv_files:
            console.print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        console.print(f"[red]Raw datasets directory not found: {raw_dir}[/red]")
        raise typer.Exit(1)

    console.print("[green]Validation complete.[/green]")


@app.command()
def stats(
    path: str = typer.Argument(
        "outputs/dataset_statistics.json",
        help="Path to dataset_statistics.json.",
    ),
) -> None:
    """Display dataset statistics from a previous pipeline run."""
    from pain_intelligence.utils.io import read_json

    if not Path(path).exists():
        console.print(f"[red]Statistics file not found: {path}[/red]")
        raise typer.Exit(1)

    data = read_json(path)

    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Documents", str(data.get("total_documents", 0)))
    table.add_row("Avg Document Length", str(data.get("average_document_length", 0)))
    table.add_row("Avg Rating", str(data.get("average_rating", 0)))
    table.add_row("Removed Documents", str(data.get("removed_documents", 0)))

    console.print(table)

    platform_dist = data.get("platform_distribution", {})
    if platform_dist:
        ptable = Table(title="Platform Distribution")
        ptable.add_column("Platform", style="cyan")
        ptable.add_column("Count", style="green")
        for platform, count in platform_dist.items():
            ptable.add_row(platform, str(count))
        console.print(ptable)


@app.command()
def dashboard(
    output: str = typer.Option(
        "knowledge/reports",
        "--output",
        "-o",
        help="Output directory for dashboard files.",
    ),
) -> None:
    """Generate pipeline quality dashboard (JSON + TXT)."""
    console.print("[bold cyan]Generating Pipeline Dashboard...[/bold cyan]")

    from pain_intelligence.pipeline.verify import generate_dashboard

    dash = generate_dashboard(output_dir=output)

    # Print summary
    table = Table(title="Pipeline Dashboard Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Documents", str(dash.get("documents", 0)))
    table.add_row("Observation Count", str(dash.get("observation_count", 0)))
    table.add_row("Evidence Count", str(dash.get("evidence_count", 0)))
    table.add_row("Problem Signals", str(dash.get("problem_signal_count", 0)))
    table.add_row("Embedding Vectors", str(dash.get("embedding_count", 0)))
    table.add_row("Relationships", str(dash.get("relationship_count", 0)))
    table.add_row("Clusters", str(dash.get("cluster_count", 0)))
    table.add_row("Pipeline Duration", str(dash.get("pipeline_duration", "N/A")))
    table.add_row("Success/Failure", dash.get("overall_status", "unknown"))
    console.print(table)

    console.print(f"[green]Dashboard written to {output}/[/green]")


@app.command()
def verify(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to YAML configuration file.",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        "-f",
        help="Attempt to fix minor issues.",
    ),
) -> None:
    """Verify pipeline integrity — checks all assets, run_ids, schemas."""
    console.print("[bold]Pipeline Integrity Verification[/bold]")

    from pain_intelligence.pipeline.verify import verify_pipeline

    report = verify_pipeline(config_path=config, fix=fix)

    overall = report.get("overall", "UNKNOWN")
    color = "green" if overall == "PASS" else "red" if overall == "FAIL" else "yellow"
    console.print(f"\n[bold {color}]Overall: {overall}[/bold {color}]")

    for check_name, check_result in report.get("checks", {}).items():
        status = check_result.get("status", "UNKNOWN")
        detail = check_result.get("detail", "")
        scolor = "green" if status == "PASS" else "red" if status == "FAIL" else "yellow"
        console.print(f"  [{scolor}]{status:8s}[/{scolor}] {check_name}: {detail}")

    warnings = report.get("warnings", [])
    if warnings:
        console.print(f"\n[yellow]Warnings ({len(warnings)}):[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]! {w}[/yellow]")

    errors = report.get("errors", [])
    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/red]")
        for e in errors:
            console.print(f"  [red]x {e}[/red]")


@app.command()
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
    """Evaluate pipeline quality across all stages."""
    console.print("[bold cyan]Pain Intelligence[/bold cyan] — Pipeline Evaluation")

    from phase2.evaluation.evaluator import EvaluationOrchestrator
    from phase2.evaluation.exporter import export_all
    from phase2.evaluation.reports import generate_summary

    orchestrator = EvaluationOrchestrator(knowledge_dir=knowledge_dir)
    result = orchestrator.evaluate()

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
        status = "PASS" if health.score >= 70 else "WARN" if health.score >= 40 else "FAIL"
        table.add_row(name, f"{health.score:.1f}/100", status)
    table.add_section()
    table.add_row("[bold]Overall", f"[bold]{result.overall_health_score:.1f}/100", "")
    console.print(table)

    if result.worst_stage:
        console.print(f"\n[yellow]Worst stage: {result.worst_stage}[/yellow]")

    if result.all_warnings:
        wt = Table(title="Warnings")
        wt.add_column("Warning")
        for w in result.all_warnings:
            wt.add_row(f"[yellow]{w}[/yellow]")
        console.print(wt)

    if result.recommendations:
        rt = Table(title="Recommendations")
        rt.add_column("Recommendation")
        for r in result.recommendations:
            rt.add_row(f"[cyan]{r}[/cyan]")
        console.print(rt)

    export_all(result, output_dir=output_dir)

    if dashboard:
        from phase2.evaluation.dashboard import generate_dashboard
        generate_dashboard(result, output_dir=output_dir)
        console.print(f"[green]Dashboard written to {output_dir}/[/green]")

    console.print(f"[green]Reports written to {output_dir}/[/green]")

    if json_output:
        import json
        s = generate_summary(result)
        console.print(json.dumps(s, indent=2, default=str))


def _print_summary(stats: dict) -> None:
    """Print a rich summary table."""
    table = Table(title="Pipeline Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Datasets Processed", str(len(stats.get("datasets_processed", []))))
    table.add_row("Total Loaded", str(stats.get("total_loaded", 0)))
    table.add_row("Total Processed", str(stats.get("total_processed", 0)))
    table.add_row("Total Removed", str(stats.get("total_removed", 0)))
    table.add_row("Elapsed Time", f"{stats.get('elapsed_seconds', 0)}s")
    table.add_row("Errors", str(len(stats.get("errors", []))))

    console.print(table)

    for ds in stats.get("datasets_processed", []):
        console.print(
            f"  [cyan]{ds['file']}[/cyan]: "
            f"loaded={ds['loaded']}, processed={ds['processed']}, removed={ds['removed']}"
        )


if __name__ == "__main__":
    app()
