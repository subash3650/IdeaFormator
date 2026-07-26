from __future__ import annotations

from typing import Any

from phase5.api.services.base import BaseEngineService


class SearchService(BaseEngineService):
    def __init__(self, knowledge_dir, config_path=None):
        super().__init__(knowledge_dir, config_path)
        self._kg_service = None
        self._opportunity_service = None
        self._trend_service = None

    async def search_all(
        self,
        query: str,
        modules: list[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        if modules is None:
            modules = ["kg", "opportunity", "trend"]

        results: dict[str, Any] = {}

        if "kg" in modules:
            from phase5.api.services.knowledge_graph import KGService
            self._kg_service = KGService(self._knowledge_dir)
            kg_results = await self._kg_service.search(query, top_k)
            results["knowledge_graph"] = kg_results

        if "opportunity" in modules:
            from phase5.api.services.opportunity import OpportunityService
            self._opportunity_service = OpportunityService(self._knowledge_dir)
            opp_results = await self._opportunity_service.search(query, top_k)
            results["opportunities"] = opp_results

        if "trend" in modules:
            from phase5.api.services.trend import TrendService
            self._trend_service = TrendService(self._knowledge_dir)
            trend_results = await self._trend_service.search(query, top_k)
            results["trends"] = trend_results

        results["total_matches"] = sum(len(v) for v in results.values() if isinstance(v, list))
        return results

    async def stats(self) -> dict[str, Any]:
        return {"available_modules": ["kg", "opportunity", "trend"]}
