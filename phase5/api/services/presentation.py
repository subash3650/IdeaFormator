from __future__ import annotations

from typing import Any

from phase5.api.services.base import BaseEngineService


class PresentationService(BaseEngineService):
    async def _get_engine(self):
        if self._engine is None:
            from phase3.presentation.engine import PresentationEngine
            from phase3.presentation.config import PresentationConfig
            config = PresentationConfig(output_dir=str(self._knowledge_dir), knowledge_dir=str(self._knowledge_dir))
            self._engine = PresentationEngine(config)
        return self._engine

    async def stats(self) -> dict[str, Any]:
        engine = await self._get_engine()
        return await self._run_in_thread(engine.stats)

    async def list_reports(self, limit: int = 20) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        reports = await self._run_in_thread(engine.list_reports, limit)
        return [r.model_dump(mode="json") for r in reports]

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        engine = await self._get_engine()
        report = await self._run_in_thread(engine.get_report, report_id)
        if report is None:
            return None
        return report.model_dump(mode="json")

    async def generate(self, report_type: str, template: str | None = None) -> dict[str, Any]:
        engine = await self._get_engine()
        result = await self._run_in_thread(engine.generate, report_type, template, None, False)
        return result

    async def search(
        self,
        query: str = "",
        report_type: str = "",
        tag: str = "",
        company: str = "",
        technology: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        engine = await self._get_engine()
        report_type_arg = report_type if report_type else None
        tag_arg = tag if tag else None
        company_arg = company if company else None
        technology_arg = technology if technology else None
        reports = await self._run_in_thread(
            engine.search, query, report_type_arg, tag_arg, company_arg, technology_arg, None, None, limit
        )
        return [r.model_dump(mode="json") for r in reports]
