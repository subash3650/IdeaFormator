from __future__ import annotations

import uvicorn
import typer

from phase5.api.app import create_app
from phase5.api.config.settings import load_api_config

api_cli = typer.Typer(help="IdeaFormator REST API")


@api_cli.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of workers"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    config = load_api_config()
    uvicorn.run(
        "phase5.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )
