from __future__ import annotations

from typing import Any

from phase5.api.services.base import BaseEngineService


class OpportunityService(BaseEngineService):
    async def _get_engine(self):
        if self._engine is None:
            from phase3.opportunity.engine import OpportunityEngine
            from phase3.opportunity.config import OpportunityConfig
            config = OpportunityConfig(output_dir=str(self._knowledge_dir), knowledge_dir=str(self._knowledge_dir))
            self._engine = OpportunityEngine(config)
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
        opportunities = await self._run_in_thread(store.load_opportunities)
        items = [o.model_dump(mode="json") for o in opportunities]
        return items[offset:offset + limit]

    async def get_by_id(self, opportunity_id: str) -> dict[str, Any] | None:
        engine = await self._get_engine()
        store = engine.store
        opportunities = await self._run_in_thread(store.load_opportunities)
        for o in opportunities:
            if o.opportunity_id == opportunity_id:
                return o.model_dump(mode="json")
        return None

    async def get_top(self, top_k: int = 10) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        store = engine.store
        opportunities = await self._run_in_thread(store.load_opportunities)
        sorted_opps = sorted(opportunities, key=lambda o: o.opportunity_score, reverse=True)
        return [o.model_dump(mode="json") for o in sorted_opps[:top_k]]
