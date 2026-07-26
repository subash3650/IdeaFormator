from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator

from fastapi import Request


async def get_knowledge_dir(request: Request) -> Path:
    settings = request.app.state.settings
    return Path(settings.knowledge_dir)


async def get_config_path(request: Request) -> Path:
    return Path("configs/default.yaml")
