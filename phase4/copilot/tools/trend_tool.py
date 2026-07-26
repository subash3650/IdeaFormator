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


@register_tool("trend")
class TrendTool(BaseTool):
    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()

    @property
    def name(self) -> str:
        return "trend"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Query detected trends with growth metrics, velocity, and momentum",
            supported_intents=[
                Intent.QUERY_TREND,
                Intent.SEARCH,
                Intent.STATISTICS,
                Intent.BRIEFING,
            ],
            priority=ToolPriority.HIGH,
            estimated_cost=2.0,
            estimated_latency_ms=200.0,
            permissions=PermissionType.READ_ONLY,
            cacheable=True,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        try:
            action = params.get("action", "search")
            store = self._get_store()
            trends = store.load_trends()

            if action == "stats":
                data = self._stats(trends)
            elif action == "growing":
                data = self._filter(trends, "growing", params)
            elif action == "declining":
                data = self._filter(trends, "declining", params)
            elif action == "emerging":
                data = self._filter(trends, "emerging", params)
            elif action == "detail":
                data = self._detail(trends, params)
            elif action == "by_company":
                data = self._by_field(trends, "affected_companies", params)
            elif action == "by_product":
                data = self._by_field(trends, "affected_products", params)
            else:
                data = self._search(trends, params)

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

    def _get_store(self) -> Any:
        from phase3.trend.store import TrendStore
        return TrendStore(self._config.phase3_dir)

    def _stats(self, trends: list) -> dict[str, Any]:
        scores = []
        type_dist: dict[str, int] = {}
        for t in trends:
            metrics = getattr(t, "metrics", None)
            if metrics and hasattr(metrics, "trend_score"):
                scores.append(metrics.trend_score)
            tt = str(getattr(t, "trend_type", ""))
            type_dist[tt] = type_dist.get(tt, 0) + 1
        return {
            "total": len(trends),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "type_distribution": type_dist,
        }

    def _filter(self, trends: list, trend_type: str, params: dict) -> dict[str, Any]:
        limit = params.get("top_k", 10)
        results = []
        for t in trends:
            if str(getattr(t, "trend_type", "")).lower() == trend_type:
                metrics = getattr(t, "metrics", None)
                results.append({
                    "trend_id": getattr(t, "trend_id", ""),
                    "title": getattr(t, "title", ""),
                    "trend_type": trend_type,
                    "score": metrics.trend_score if metrics else 0,
                    "growth_pct": metrics.growth_pct if metrics else 0,
                    "confidence": metrics.confidence if metrics else 0,
                })
        results.sort(key=lambda r: r["score"], reverse=True)
        return {"trend_type": trend_type, "total": len(results), "results": results[:limit]}

    def _detail(self, trends: list, params: dict) -> dict[str, Any]:
        trend_id = params.get("trend_id", "")
        for t in trends:
            if getattr(t, "trend_id", "") == trend_id:
                metrics = getattr(t, "metrics", None)
                return {
                    "trend_id": trend_id,
                    "title": getattr(t, "title", ""),
                    "summary": getattr(t, "summary", ""),
                    "trend_type": str(getattr(t, "trend_type", "")),
                    "direction": str(getattr(t, "trend_direction", "")),
                    "subject": str(getattr(t, "trend_subject", "")),
                    "metrics": {
                        "growth_pct": metrics.growth_pct if metrics else 0,
                        "velocity": metrics.velocity if metrics else 0,
                        "momentum": metrics.momentum if metrics else 0,
                        "confidence": metrics.confidence if metrics else 0,
                        "trend_score": metrics.trend_score if metrics else 0,
                        "duration_days": metrics.duration_days if metrics else 0,
                    } if metrics else {},
                    "companies": getattr(t, "affected_companies", []),
                    "products": getattr(t, "affected_products", []),
                    "technologies": getattr(t, "affected_technologies", []),
                }
        return {"error": f"Trend not found: {trend_id}"}

    def _by_field(self, trends: list, field: str, params: dict) -> dict[str, Any]:
        value = (params.get("value", "") or "").lower()
        limit = params.get("top_k", 10)
        results = []
        for t in trends:
            items = [str(x).lower() for x in getattr(t, field, [])]
            if value in items:
                metrics = getattr(t, "metrics", None)
                results.append({
                    "trend_id": getattr(t, "trend_id", ""),
                    "title": getattr(t, "title", ""),
                    "score": metrics.trend_score if metrics else 0,
                })
        results.sort(key=lambda r: r["score"], reverse=True)
        return {"field": field, "value": value, "total": len(results), "results": results[:limit]}

    def _search(self, trends: list, params: dict) -> dict[str, Any]:
        query = (params.get("query", "") or "").lower()
        limit = params.get("top_k", 10)
        scored = []
        for t in trends:
            title = (getattr(t, "title", "") or "").lower()
            summary = (getattr(t, "summary", "") or "").lower()
            companies = " ".join(getattr(t, "affected_companies", [])).lower()

            score = 0
            if query in title:
                score += 3
            if query in summary:
                score += 2
            if query in companies:
                score += 1

            if score > 0:
                metrics = getattr(t, "metrics", None)
                scored.append((score, {
                    "trend_id": getattr(t, "trend_id", ""),
                    "title": getattr(t, "title", ""),
                    "score": metrics.trend_score if metrics else 0,
                    "trend_type": str(getattr(t, "trend_type", "")),
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return {
            "query": params.get("query", ""),
            "total": len(scored),
            "results": [s[1] for s in scored[:limit]],
        }

    def _build_citations(self, data: dict) -> list[Citation]:
        citations: list[Citation] = []
        for result in data.get("results", []):
            tid = result.get("trend_id", "")
            if tid:
                citations.append(Citation(
                    source_module=CitationSource.TREND,
                    source_id=tid,
                    source_title=result.get("title", ""),
                    confidence=result.get("score", 0),
                    snippet=f"{result.get('trend_type', '')}: score={result.get('score', 0):.2f}",
                ))
        return citations
