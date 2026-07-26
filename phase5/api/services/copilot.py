from __future__ import annotations

from typing import Any, AsyncGenerator

from phase5.api.services.base import BaseEngineService


class CopilotService(BaseEngineService):
    def __init__(self, knowledge_dir, config_path=None):
        super().__init__(knowledge_dir, config_path)
        self._engine = None

    async def _get_engine(self):
        if self._engine is None:
            from phase4.copilot.engine import CopilotEngine
            from phase4.copilot.config import CopilotConfig
            config = CopilotConfig(knowledge_dir=self._knowledge_dir)
            self._engine = CopilotEngine(config)
        return self._engine

    async def chat(self, query: str, session_id: str | None = None) -> dict[str, Any]:
        engine = await self._get_engine()
        response = await self._run_in_thread(engine.chat, query, session_id)
        return response.model_dump(mode="json")

    async def stream(
        self, query: str, session_id: str | None = None
    ) -> AsyncGenerator[str, None]:
        engine = await self._get_engine()
        chunks = await self._run_in_thread(lambda: list(engine.chat_stream(query, session_id)))

        from phase5.api.responses.streaming import SSEStreamer
        for chunk in chunks:
            if chunk.chunk_type == "token":
                yield SSEStreamer.format_token(chunk.data, chunk.index)
            elif chunk.chunk_type == "citations":
                yield SSEStreamer.format_citations(chunk.data if isinstance(chunk.data, list) else [])
            elif chunk.final:
                yield SSEStreamer.format_done(chunk.data if isinstance(chunk.data, str) else session_id or "")

    async def new_session(self) -> str:
        engine = await self._get_engine()
        return await self._run_in_thread(engine.new_session)

    async def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        messages = await self._run_in_thread(engine.get_session_history, session_id)
        return [m.model_dump(mode="json") for m in messages]

    async def stats(self) -> dict[str, Any]:
        engine = await self._get_engine()
        return await self._run_in_thread(engine.stats)
