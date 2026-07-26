from __future__ import annotations

from pathlib import Path
from typing import Any

from phase5.api.services.base import BaseEngineService


class ExportService(BaseEngineService):
    async def export_opportunities(self, format: str = "json", output_dir: str | None = None) -> dict[str, Any]:
        from phase3.opportunity.store import OpportunityStore
        from phase3.opportunity.exporter import OpportunityExporter
        store = OpportunityStore(self._knowledge_dir / "opportunity")
        exporter = OpportunityExporter(store)
        return await self._run_in_thread(self._export_opps, exporter, format, output_dir)

    def _export_opps(self, exporter, format: str, output_dir: str | None) -> dict[str, Any]:
        if format == "report":
            p = exporter.export_report()
        elif format == "csv":
            p = exporter.export_csv()
        elif format == "summary":
            p = exporter.export_summary()
        elif format == "dashboard":
            p = exporter.export_dashboard()
        else:
            p = exporter.export_report()
        return {"path": str(p), "format": format}

    async def export_trends(self, format: str = "json", output_dir: str | None = None) -> dict[str, Any]:
        from phase3.trend.store import TrendStore
        from phase3.trend.exporter import TrendExporter
        store = TrendStore(self._knowledge_dir / "trend")
        exporter = TrendExporter(store)
        return await self._run_in_thread(self._export_trends, exporter, format, output_dir)

    def _export_trends(self, exporter, format: str, output_dir: str | None) -> dict[str, Any]:
        if format == "report":
            p = exporter.export_report()
        elif format == "csv":
            p = exporter.export_csv()
        elif format == "summary":
            p = exporter.export_summary()
        elif format == "dashboard":
            p = exporter.export_dashboard()
        else:
            p = exporter.export_report()
        return {"path": str(p), "format": format}

    async def export_reports(self, report_id: str | None = None, format: str = "json") -> dict[str, Any]:
        from phase3.presentation.exporter import PresentationExporter
        from phase3.presentation.config import PresentationConfig
        config = PresentationConfig(output_dir=str(self._knowledge_dir / "presentation"), knowledge_dir=str(self._knowledge_dir))
        exporter = PresentationExporter(config)
        return await self._run_in_thread(self._export_reports, exporter, report_id, format)

    def _export_reports(self, exporter, report_id: str | None, format: str) -> dict[str, Any]:
        if report_id:
            return {"report_id": report_id, "format": format, "exported": True}
        result = exporter.export_all()
        return {"count": len(result), "format": format, "exported": True}

    async def stats(self) -> dict[str, Any]:
        return {
            "available_export_formats": {
                "opportunities": ["report", "csv", "summary", "dashboard", "json"],
                "trends": ["report", "csv", "summary", "dashboard", "json"],
                "reports": ["json", "markdown", "pdf"],
            }
        }
