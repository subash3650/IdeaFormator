from __future__ import annotations

from pathlib import Path
from typing import Any

from phase5.api.services.base import BaseEngineService


class KGService(BaseEngineService):
    async def _get_engine(self):
        if self._engine is None:
            from phase2.knowledge_graph.engine import KnowledgeGraphEngine
            from phase2.knowledge_graph.config import KnowledgeGraphConfig
            config = KnowledgeGraphConfig(output_dir=str(self._knowledge_dir))
            self._engine = KnowledgeGraphEngine(config)
        return self._engine

    async def stats(self) -> dict[str, Any]:
        engine = await self._get_engine()
        return await self._run_in_thread(engine.stats)

    async def search(self, query: str, top_k: int = 10, type_filter: str = "") -> list[dict[str, Any]]:
        engine = await self._get_engine()
        return await self._run_in_thread(engine.search, query, top_k, type_filter)

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        results = await self.search(node_id, top_k=1)
        for r in results:
            if r.get("node_id") == node_id or r.get("id") == node_id:
                return r
        return None

    async def get_stats_summary(self) -> dict[str, Any]:
        return await self.stats()
