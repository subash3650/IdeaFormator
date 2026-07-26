from __future__ import annotations

from typing import Any

from phase5.api.services.base import BaseEngineService


class ReasoningService(BaseEngineService):
    async def _get_engine(self):
        if self._engine is None:
            from phase2.reasoning.engine import ReasoningEngine
            from phase2.reasoning.config import ReasoningConfig
            config = ReasoningConfig(output_dir=str(self._knowledge_dir), knowledge_dir=str(self._knowledge_dir))
            self._engine = ReasoningEngine(config)
        return self._engine

    async def stats(self) -> dict[str, Any]:
        engine = await self._get_engine()
        return await self._run_in_thread(engine.stats)

    async def get_inferences(self) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        store = engine.store
        inferences = await self._run_in_thread(store.load_inferences)
        return [i.model_dump(mode="json") for i in inferences]

    async def get_chains(self) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        store = engine.store
        chains = await self._run_in_thread(store.load_chains)
        return [c.model_dump(mode="json") for c in chains]

    async def get_root_causes(self) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        store = engine.store
        causes = await self._run_in_thread(store.load_root_causes)
        return [c.model_dump(mode="json") for c in causes]

    async def get_evidence(self) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        store = engine.store
        evidence = await self._run_in_thread(store.load_evidence_aggregations)
        return [e.model_dump(mode="json") for e in evidence]
