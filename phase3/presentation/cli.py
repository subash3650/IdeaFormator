from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from phase3.presentation.engine import PresentationEngine
from phase3.presentation.schema import ReportFormat

console = Console()
presentation_app = typer.Typer(
    name="presentation",
    help="Generate and manage intelligence reports",
    no_args_is_help=True,
)


def _engine() -> PresentationEngine:
    return PresentationEngine()


@presentation_app.command()
def generate(
    report_type: str = typer.Option("executive_summary", "--type", "-t", help="Report type"),
    template: str = typer.Option(None, "--template", help="Template name"),
    formats: str = typer.Option("json,markdown,html,csv", "--formats", "-f", help="Comma-separated output formats"),
):
    fmt_list = [ReportFormat(f.strip()) for f in formats.split(",") if f.strip()]
    result = _engine().generate(
        report_type=report_type,
        template_name=template,
        output_formats=fmt_list,
    )
    console.print("[green]Report generated successfully[/green]")
    _print_result(result)


@presentation_app.command()
def list_reports(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of reports to show"),
):
    reports = _engine().list_reports(limit=limit)
    if not reports:
        console.print("[yellow]No reports found[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Reports (last {len(reports)})")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Title")
    table.add_column("Sections")
    table.add_column("Formats")
    table.add_column("Date")

    for r in reports:
        table.add_row(
            r.report_id[:12],
            r.report_type.value,
            r.title[:40],
            str(r.sections_count),
            ", ".join(f.value for f in r.formats)[:20],
            r.generated_at[:10],
        )

    console.print(table)


@presentation_app.command()
def show(
    report_id: str = typer.Argument(..., help="Report ID"),
):
    model = _engine().get_report(report_id)
    if model is None:
        console.print(f"[red]Report not found: {report_id}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Report:[/bold] {model.title}")
    console.print(f"[bold]Type:[/bold] {model.report_type.value}")
    console.print(f"[bold]Generated:[/bold] {model.generated_at}")
    console.print(f"[bold]Sections:[/bold] {len(model.sections)}")
    console.print(f"[bold]Tags:[/bold] {', '.join(model.tags) if model.tags else 'none'}")
    console.print(f"[bold]Companies:[/bold] {', '.join(model.companies) if model.companies else 'none'}")
    console.print(f"[bold]Technologies:[/bold] {', '.join(model.technologies) if model.technologies else 'none'}")
    console.print()

    sections_table = Table(title="Sections")
    sections_table.add_column("#", style="dim")
    sections_table.add_column("Type")
    sections_table.add_column("Title")

    for i, s in enumerate(model.sections, 1):
        sections_table.add_row(str(i), s.section_type.value, s.title)

    console.print(sections_table)

    if model.summaries.one_paragraph:
        console.print(f"\n[bold]Summary:[/bold] {model.summaries.one_paragraph}")


@presentation_app.command()
def export(
    report_id: str = typer.Argument(..., help="Report ID"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory"),
    formats: str = typer.Option("json,markdown,html,csv", "--formats", "-f", help="Comma-separated output formats"),
):
    fmt_list = [ReportFormat(f.strip()) for f in formats.split(",") if f.strip()]
    result = _engine().export(
        report_id=report_id,
        output_dir=Path(output_dir) if output_dir else None,
        formats=fmt_list,
    )

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"Exported: {report_id}")
    table.add_column("Format", style="green")
    table.add_column("Path")

    for fmt, path in result.items():
        if path:
            table.add_row(fmt, path)
        else:
            table.add_row(fmt, "[red]failed[/red]")

    console.print(table)


@presentation_app.command()
def stats():
    data = _engine().stats()
    if data["total_reports"] == 0:
        console.print("[yellow]No reports in store[/yellow]")
        raise typer.Exit()

    table = Table(title="Presentation Store Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Total Reports", str(data["total_reports"]))
    table.add_row("Avg Sections", str(data["avg_sections"]))
    table.add_row("Avg Charts", str(data["avg_charts"]))
    table.add_row("Avg Elapsed (s)", str(data["avg_elapsed"]))

    for fmt, count in data.get("by_format", {}).items():
        table.add_row(f"  Format: {fmt}", str(count))

    for rtype, count in data.get("by_type", {}).items():
        table.add_row(f"  Type: {rtype}", str(count))

    table.add_row("Earliest", data["earliest"] or "-")
    table.add_row("Latest", data["latest"] or "-")

    console.print(table)


@presentation_app.command()
def search(
    query: str = typer.Option(None, "--query", "-q", help="Search query"),
    report_type: str = typer.Option(None, "--type", "-t", help="Filter by report type"),
    tag: str = typer.Option(None, "--tag", help="Filter by tag"),
    company: str = typer.Option(None, "--company", "-c", help="Filter by company"),
    technology: str = typer.Option(None, "--tech", help="Filter by technology"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max results"),
):
    results = _engine().search(
        query=query,
        report_type=report_type,
        tag=tag,
        company=company,
        technology=technology,
        limit=limit,
    )

    if not results:
        console.print("[yellow]No matching reports[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Search Results ({len(results)})")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Title")
    table.add_column("Date")

    for r in results:
        table.add_row(r.report_id[:12], r.report_type.value, r.title[:50], r.generated_at[:10])

    console.print(table)


def _print_result(result: dict) -> None:
    console.print(f"  Report ID:  [cyan]{result['report_id']}[/cyan]")
    console.print(f"  Title:      {result['title']}")
    console.print(f"  Sections:   {result['sections_count']}")
    console.print(f"  Charts:     {result['charts_count']}")
    console.print(f"  Formats:    {', '.join(result['formats'])}")
    console.print(f"  Elapsed:    {result['elapsed_seconds']}s")
