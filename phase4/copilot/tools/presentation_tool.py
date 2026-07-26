from __future__ import annotations

import time
from typing import Any

from phase4.copilot.tools.base import BaseTool
from phase4.copilot.tools.registry import register_tool
from phase4.copilot.schema import (
    Citation,
    CitationSource,
    Intent,
    PermissionType,
    ToolMetadata,
    ToolPriority,
    ToolResult,
)
from phase4.copilot.config import CopilotConfig


@register_tool("presentation")
class PresentationTool(BaseTool):
    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()

    @property
    def name(self) -> str:
        return "presentation"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Access previously generated reports and generate new intelligence briefings",
            supported_intents=[
                Intent.QUERY_PRESENTATION,
                Intent.SEARCH,
                Intent.BRIEFING,
                Intent.STATISTICS,
            ],
            priority=ToolPriority.MEDIUM,
            estimated_cost=3.0,
            estimated_latency_ms=500.0,
            permissions=PermissionType.READ_ONLY,
            cacheable=True,
            dependencies=["opportunity", "trend"],
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        try:
            action = params.get("action", "list")
            engine = self._get_engine()

            if action == "list":
                data = self._list(engine, params)
            elif action == "get":
                data = self._get(engine, params)
            elif action == "generate":
                data = self._generate(engine, params)
            elif action == "stats":
                data = engine.stats()
            else:
                data = self._list(engine, params)

            citations = self._build_citations(data)

            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
                citations=citations,
                elapsed_ms=round(elapsed, 1),
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                elapsed_ms=round(elapsed, 1),
                error=str(e),
            )

    def _get_engine(self) -> Any:
        from phase3.presentation.engine import PresentationEngine
        return PresentationEngine(config=self._config)

    def _list(self, engine: Any, params: dict) -> dict[str, Any]:
        limit = params.get("top_k", 20)
        reports = engine.list_reports(limit=limit)
        return {
            "total": len(reports),
            "reports": [
                {
                    "report_id": r.report_id,
                    "title": r.title,
                    "type": str(r.report_type),
                    "sections": r.sections_count,
                    "formats": [str(f) for f in r.formats],
                    "generated_at": r.generated_at,
                }
                for r in reports
            ],
        }

    def _get(self, engine: Any, params: dict) -> dict[str, Any]:
        report_id = params.get("report_id", "")
        model = engine.get_report(report_id)
        if model is None:
            return {"error": f"Report not found: {report_id}"}
        return {
            "report_id": model.report_id,
            "title": model.title,
            "type": str(model.report_type),
            "subtitle": model.subtitle,
            "generated_at": model.generated_at,
            "sections": [
                {
                    "section_type": str(s.section_type),
                    "title": s.title,
                    "content": s.content,
                }
                for s in model.sections
            ],
            "tags": model.tags,
            "companies": model.companies,
            "technologies": model.technologies,
            "products": model.products,
            "summaries": {
                "one_paragraph": model.summaries.one_paragraph,
                "one_sentence": model.summaries.one_sentence,
            },
        }

    def _generate(self, engine: Any, params: dict) -> dict[str, Any]:
        report_type = params.get("report_type", "executive_summary")
        template = params.get("template")
        result = engine.generate(
            report_type=report_type,
            template_name=template,
            force=params.get("force", False),
        )
        return result

    def _build_citations(self, data: dict) -> list[Citation]:
        citations: list[Citation] = []
        for r in data.get("reports", []):
            rid = r.get("report_id", "")
            if rid:
                citations.append(Citation(
                    source_module=CitationSource.PRESENTATION,
                    source_id=rid,
                    source_title=r.get("title", ""),
                    snippet=f"Report: {r.get('type', '')}",
                ))
        return citations
