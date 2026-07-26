from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from phase4.copilot.engine import CopilotEngine
from phase4.copilot.schema import ResponseFormat

console = Console()
copilot_app = typer.Typer(
    name="copilot",
    help="AI Business Copilot — intelligence query interface",
    no_args_is_help=True,
)


def _engine() -> CopilotEngine:
    return CopilotEngine()


@copilot_app.command()
def chat(
    query: str = typer.Argument(..., help="Your question"),
    session: str = typer.Option(None, "--session", "-s", help="Session ID for multi-turn conversation"),
    stream: bool = typer.Option(False, "--stream", help="Stream response tokens"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    fmt = ResponseFormat.JSON if json_output else ResponseFormat.MARKDOWN
    engine = _engine()

    if stream:
        for chunk in engine.chat_stream(query, session_id=session):
            if chunk.chunk_type == "token":
                console.print(chunk.data, end="")
            elif chunk.chunk_type == "citations":
                console.print(chunk.data)
            elif chunk.chunk_type == "done":
                console.print("\n[dim]Stream complete[/dim]")
        return

    response = engine.chat(query, session_id=session, response_format=fmt)
    console.print(response.content)

    if response.suggested_followups:
        console.print("\n[bold cyan]Try asking:[/bold cyan]")
        for f in response.suggested_followups:
            console.print(f"  [dim]• {f}[/dim]")

    if response.citations:
        console.print(f"\n[dim]{len(response.citations)} citations[/dim]")


@copilot_app.command()
def ask(
    query: str = typer.Argument(..., help="Your question"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    fmt = ResponseFormat.JSON if json_output else ResponseFormat.MARKDOWN
    response = _engine().ask(query, response_format=fmt)
    console.print(response.content)


@copilot_app.command()
def sessions():
    active = _engine().session_manager.list_active()
    if not active:
        console.print("[yellow]No active sessions[/yellow]")
        return

    table = Table(title="Active Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Messages")
    table.add_column("Last Active")
    table.add_column("Status")

    for s in active:
        table.add_row(
            s.session_id[:16],
            str(s.message_count),
            s.last_active_at[:19],
            s.status.value,
        )
    console.print(table)


@copilot_app.command()
def history(
    session_id: str = typer.Argument(..., help="Session ID"),
):
    messages = _engine().get_session_history(session_id)
    if not messages:
        console.print(f"[yellow]No history for session: {session_id}[/yellow]")
        return

    for msg in messages:
        role_color = "green" if msg.role.value == "user" else "blue"
        console.print(f"[{role_color}]{msg.role.value.upper()}[/{role_color}]")
        console.print(f"  {msg.content[:200]}")
        console.print()


@copilot_app.command()
def stats():
    data = _engine().stats()
    table = Table(title="Copilot Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    for k, v in data.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)
