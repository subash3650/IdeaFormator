"""Typer CLI for the Pain Intelligence Engine."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pain_intelligence.pipeline.orchestrator import Orchestrator
from phase2.embeddings.cli import embeddings_app
from phase2.similarity.cli import similarity_app

app = typer.Typer(
    name="pain-intelligence",
    help="Pain Intelligence Engine - Data Ingestion & Preprocessing Pipeline",
    add_completion=False,
)
app.add_typer(embeddings_app)
app.add_typer(similarity_app)
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
        "outputs/processed.parquet",
        "--data",
        "-d",
        help="Path to processed dataset.",
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
    table.add_row("Observations", str(result.get("observations_count", 0)))
    table.add_row("Evidence Records", str(result.get("evidence_count", 0)))
    table.add_row("Problem Signals", str(result.get("signal_count", 0)))
    table.add_row("Elapsed Time", f"{result.get('elapsed_seconds', 0)}s")
    console.print(table)


@app.command()
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
