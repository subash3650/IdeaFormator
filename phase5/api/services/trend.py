from __future__ import annotations

from typing import Any

from phase5.api.services.base import BaseEngineService


class TrendService(BaseEngineService):
    async def _get_engine(self):
        if self._engine is None:
            from phase3.trend.engine import TrendEngine
            from phase3.trend.config import TrendConfig
            config = TrendConfig(output_dir=str(self._knowledge_dir), knowledge_dir=str(self._knowledge_dir))
            self._engine = TrendEngine(config)
        return self._engine

    async def stats(self) -> dict[str, Any]:
        engine = await self._get_engine()
        return await self._run_in_thread(engine.stats)

    async def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        return await self._run_in_thread(engine.search, query, top_k)

    async def list_all(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        store = engine.store
        trends = await self._run_in_thread(store.load_trends)
        items = [t.model_dump(mode="json") for t in trends]
        return items[offset:offset + limit]

    async def get_by_id(self, trend_id: str) -> dict[str, Any] | None:
        engine = await self._get_engine()
        store = engine.store
        trends = await self._run_in_thread(store.load_trends)
        for t in trends:
            if t.trend_id == trend_id:
                return t.model_dump(mode="json")
        return None

    async def growing(self, top_k: int = 10) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        store = engine.store
        trends = await self._run_in_thread(store.load_trends)
        growing = [t for t in trends if t.trend_type == "growing"]
        sorted_g = sorted(growing, key=lambda t: t.metrics.growth_pct if hasattr(t, "metrics") else 0, reverse=True)
        return [t.model_dump(mode="json") for t in sorted_g[:top_k]]

    async def emerging(self, top_k: int = 10) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        store = engine.store
        trends = await self._run_in_thread(store.load_trends)
        emerging = [t for t in trends if t.trend_type == "emerging"]
        return [t.model_dump(mode="json") for t in emerging[:top_k]]
