from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseEngineService:
    _executor = ThreadPoolExecutor(max_workers=4)

    def __init__(self, knowledge_dir: Path, config_path: Path | None = None) -> None:
        self._knowledge_dir = knowledge_dir
        self._config_path = config_path
        self._engine: Any = None

    async def _run_in_thread(self, func, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    async def stats(self) -> dict[str, Any]:
        raise NotImplementedError

    async def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        raise NotImplementedError
