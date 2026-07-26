"""Typer CLI sub-application for the Ingestion Framework."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pain_intelligence.ingestion.config import load_ingestion_config
from pain_intelligence.ingestion.engine import IngestionEngine

ingest_app = typer.Typer(
    name="ingest",
    help="Data Ingestion Framework — collect, normalize, and store data from multiple platforms.",
    no_args_is_help=True,
)
console = Console()


def _get_engine(config: str) -> IngestionEngine:
    """Load config and create an IngestionEngine."""
    if not Path(config).exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)
    return IngestionEngine(config)


@ingest_app.command()
def run(
    config: str = typer.Option(
        "configs/ingestion.yaml",
        "--config",
        "-c",
        help="Path to ingestion YAML configuration.",
    ),
    sources: str = typer.Option(
        None,
        "--sources",
        "-s",
        help="Comma-separated list of sources to run (e.g. 'github,hackernews'). Default: all enabled.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-fetch ignoring existing state.",
    ),
) -> None:
    """Run all enabled ingestion collectors."""
    console.print("[bold green]Ingestion Framework[/bold green] v0.1.0")
    console.print(f"Config: {config}")

    engine = _get_engine(config)
    source_list = [s.strip() for s in sources.split(",")] if sources else None
    result = engine.run(sources=source_list, force=force)

    _print_summary(result)


@ingest_app.command()
def github(
    config: str = typer.Option(
        "configs/ingestion.yaml",
        "--config",
        "-c",
        help="Path to ingestion YAML configuration.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-fetch."),
) -> None:
    """Run the GitHub collector only."""
    console.print("[bold cyan]GitHub Collector[/bold cyan]")
    engine = _get_engine(config)
    result = engine.run_collector("github", force=force)
    _print_single_result("github", result)


@ingest_app.command()
def hackernews(
    config: str = typer.Option(
        "configs/ingestion.yaml",
        "--config",
        "-c",
        help="Path to ingestion YAML configuration.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-fetch."),
) -> None:
    """Run the Hacker News collector only."""
    console.print("[bold cyan]Hacker News Collector[/bold cyan]")
    engine = _get_engine(config)
    result = engine.run_collector("hackernews", force=force)
    _print_single_result("hackernews", result)


@ingest_app.command()
def producthunt(
    config: str = typer.Option(
        "configs/ingestion.yaml",
        "--config",
        "-c",
        help="Path to ingestion YAML configuration.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-fetch."),
) -> None:
    """Run the Product Hunt collector only."""
    console.print("[bold cyan]Product Hunt Collector[/bold cyan]")
    engine = _get_engine(config)
    result = engine.run_collector("producthunt", force=force)
    _print_single_result("producthunt", result)


@ingest_app.command()
def youtube(
    config: str = typer.Option(
        "configs/ingestion.yaml",
        "--config",
        "-c",
        help="Path to ingestion YAML configuration.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-fetch."),
) -> None:
    """Run the YouTube collector only."""
    console.print("[bold cyan]YouTube Collector[/bold cyan]")
    engine = _get_engine(config)
    result = engine.run_collector("youtube", force=force)
    _print_single_result("youtube", result)


@ingest_app.command()
def playstore(
    config: str = typer.Option(
        "configs/ingestion.yaml",
        "--config",
        "-c",
        help="Path to ingestion YAML configuration.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-fetch."),
    package: str | None = typer.Option(
        None, "--package", "-p", help="Single package to collect."
    ),
    category: str | None = typer.Option(
        None, "--category", help="Filter apps by category from playstore_apps.yaml."
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Max reviews per app."
    ),
    language: str | None = typer.Option(
        None, "--language", help="Language code (e.g., en, hi, ja)."
    ),
    country: str | None = typer.Option(
        None, "--country", help="Country code (e.g., us, in, jp)."
    ),
) -> None:
    """Run the Google Play Store public review collector."""
    console.print("[bold cyan]Google Play Store Collector[/bold cyan]")
    engine = _get_engine(config)

    # Override collector config with CLI options
    cfg = engine._config.collectors.get("playstore")
    if cfg and (package or category or limit or language or country):
        from pain_intelligence.ingestion.config import CollectorConfig
        new_apps = list(cfg.apps) if cfg.apps else []
        new_config_path = cfg.apps_config_path
        if package:
            new_apps = [package]
            new_config_path = None
        elif category:
            new_config_path = cfg.apps_config_path or "configs/playstore_apps.yaml"

        updated_cfg = CollectorConfig(
            enabled=cfg.enabled,
            api_key_env=cfg.api_key_env,
            batch_size=cfg.batch_size,
            max_pages=cfg.max_pages,
            rate_limit=cfg.rate_limit,
            timeout=cfg.timeout,
            retry_count=cfg.retry_count,
            retry_delay=cfg.retry_delay,
            api_version=cfg.api_version,
            language=language or cfg.language,
            country=country or cfg.country,
            review_limit=limit or cfg.review_limit,
            sort=cfg.sort,
            apps=new_apps,
            apps_config_path=new_config_path,
        )
        engine._config.collectors["playstore"] = updated_cfg

    result = engine.run_collector("playstore", force=force)
    _print_single_result("playstore", result)


@ingest_app.command()
def stats(
    config: str = typer.Option(
        "configs/ingestion.yaml",
        "--config",
        "-c",
        help="Path to ingestion YAML configuration.",
    ),
) -> None:
    """Display ingestion statistics for all sources."""
    console.print("[bold]Ingestion Statistics[/bold]")
    engine = _get_engine(config)
    stats_data = engine.stats()

    table = Table(title="Ingestion Stats")
    table.add_column("Source", style="cyan")
    table.add_column("Last Sync", style="green")
    table.add_column("Total", style="yellow")
    table.add_column("Failures", style="red")
    table.add_column("JSONL Files", style="blue")
    table.add_column("Parquet Files", style="blue")

    for source, info in stats_data.items():
        table.add_row(
            source,
            str(info.get("last_sync", "Never")),
            str(info.get("total_collected", 0)),
            str(info.get("failure_count", 0)),
            str(info.get("raw_files", 0)),
            str(info.get("parquet_files", 0)),
        )

    console.print(table)


@ingest_app.command()
def verify(
    config: str = typer.Option(
        "configs/ingestion.yaml",
        "--config",
        "-c",
        help="Path to ingestion YAML configuration.",
    ),
) -> None:
    """Verify integrity of stored ingestion data."""
    console.print("[bold]Verifying ingestion data...[/bold]")
    engine = _get_engine(config)
    results = engine.verify()

    table = Table(title="Verification Results")
    table.add_column("Source", style="cyan")
    table.add_column("Raw Dir", style="green")
    table.add_column("JSONL", style="yellow")
    table.add_column("Parquet", style="yellow")
    table.add_column("State OK", style="green")

    for source, info in results.items():
        state_ok = "Yes" if info.get("state_loaded") and not info.get("failure_count") else "No"
        table.add_row(
            source,
            "Yes" if info.get("raw_dir_exists") else "No",
            str(info.get("raw_files", 0)),
            str(info.get("parquet_files", 0)),
            state_ok,
        )

    console.print(table)


def _print_summary(result: dict) -> None:
    """Print a rich summary table for a full run."""
    table = Table(title="Ingestion Summary")
    table.add_column("Source", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Documents", style="yellow")
    table.add_column("Invalid", style="red")
    table.add_column("Pages", style="blue")
    table.add_column("Time", style="dim")

    for source, info in result.get("sources", {}).items():
        if isinstance(info, dict):
            table.add_row(
                source,
                info.get("status", "unknown"),
                str(info.get("documents_collected", 0)),
                str(info.get("documents_invalid", 0)),
                str(info.get("pages_fetched", 0)),
                f"{info.get('elapsed_seconds', 0):.1f}s",
            )

    console.print(table)
    console.print(f"Total time: {result.get('elapsed_seconds', 0):.1f}s")


def _print_single_result(source: str, result: dict) -> None:
    """Print a rich summary for a single collector run."""
    table = Table(title=f"{source.title()} Collection Result")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Status", result.get("status", "unknown"))
    table.add_row("Documents Collected", str(result.get("documents_collected", 0)))
    table.add_row("Documents Invalid", str(result.get("documents_invalid", 0)))
    table.add_row("Pages Fetched", str(result.get("pages_fetched", 0)))
    table.add_row("API Calls", str(result.get("api_calls", 0)))
    table.add_row("Elapsed Time", f"{result.get('elapsed_seconds', 0):.1f}s")

    if result.get("error"):
        table.add_row("Error", result["error"])

    console.print(table)


if __name__ == "__main__":
    ingest_app()
