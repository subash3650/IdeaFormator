from __future__ import annotations

import time
from typing import Any

from phase4.copilot.tools.base import BaseTool
from phase4.copilot.tools.registry import register_tool
from phase4.copilot.schema import (
    Citation,
    CitationSource,
    ComparisonEntityType,
    Intent,
    PermissionType,
    ToolMetadata,
    ToolPriority,
    ToolResult,
)
from phase4.copilot.config import CopilotConfig


@register_tool("comparison")
class ComparisonTool(BaseTool):
    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()

    @property
    def name(self) -> str:
        return "comparison"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Compare two entities (opportunities, trends, companies, or products)",
            supported_intents=[
                Intent.COMPARE,
            ],
            priority=ToolPriority.MEDIUM,
            estimated_cost=5.0,
            estimated_latency_ms=600.0,
            dependencies=["opportunity", "trend"],
            permissions=PermissionType.READ_ONLY,
            cacheable=False,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        try:
            entity_type_str = params.get("entity_type", "opportunity")
            try:
                entity_type = ComparisonEntityType(entity_type_str)
            except ValueError:
                entity_type = ComparisonEntityType.OPPORTUNITY

            a_id = params.get("entity_a", params.get("a", ""))
            b_id = params.get("entity_b", params.get("b", ""))

            if entity_type == ComparisonEntityType.OPPORTUNITY:
                data = self._compare_opportunities(a_id, b_id)
            elif entity_type == ComparisonEntityType.TREND:
                data = self._compare_trends(a_id, b_id)
            elif entity_type == ComparisonEntityType.COMPANY:
                data = self._compare_companies(a_id, b_id)
            elif entity_type == ComparisonEntityType.PRODUCT:
                data = self._compare_products(a_id, b_id)
            else:
                data = {"error": f"Unknown entity type: {entity_type}"}

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

    def _compare_opportunities(self, a_id: str, b_id: str) -> dict[str, Any]:
        from phase3.opportunity.store import OpportunityStore
        store = OpportunityStore(self._config.phase3_dir)
        opps = store.load_opportunities()

        a = self._find_opp(opps, a_id)
        b = self._find_opp(opps, b_id)
        if a is None or b is None:
            missing = a_id if a is None else b_id
            return {"error": f"Opportunity not found: {missing}"}

        return {
            "entity_type": "opportunity",
            "comparison": [
                {
                    "field": "title",
                    "a": getattr(a, "title", ""),
                    "b": getattr(b, "title", ""),
                },
                {
                    "field": "score",
                    "a": getattr(a, "opportunity_score", 0),
                    "b": getattr(b, "opportunity_score", 0),
                    "delta": getattr(b, "opportunity_score", 0) - getattr(a, "opportunity_score", 0),
                },
                {
                    "field": "recommendation",
                    "a": str(getattr(a, "recommendation_type", "")),
                    "b": str(getattr(b, "recommendation_type", "")),
                },
                {
                    "field": "business_model",
                    "a": str(getattr(a, "suggested_business_model", "")),
                    "b": str(getattr(b, "suggested_business_model", "")),
                },
                {
                    "field": "companies",
                    "a": getattr(a, "affected_companies", []),
                    "b": getattr(b, "affected_companies", []),
                },
                {
                    "field": "technologies",
                    "a": getattr(a, "affected_technologies", []),
                    "b": getattr(b, "affected_technologies", []),
                },
                {
                    "field": "pain_severity",
                    "a": getattr(a, "pain_severity", 0),
                    "b": getattr(b, "pain_severity", 0),
                },
                {
                    "field": "market_size",
                    "a": str(getattr(a, "estimated_market_size", "")),
                    "b": str(getattr(b, "estimated_market_size", "")),
                },
            ],
            "a_id": a_id,
            "b_id": b_id,
        }

    def _compare_trends(self, a_id: str, b_id: str) -> dict[str, Any]:
        from phase3.trend.store import TrendStore
        store = TrendStore(self._config.phase3_dir)
        trends = store.load_trends()

        a = self._find_trend(trends, a_id)
        b = self._find_trend(trends, b_id)
        if a is None or b is None:
            missing = a_id if a is None else b_id
            return {"error": f"Trend not found: {missing}"}

        a_m = getattr(a, "metrics", None) or {}
        b_m = getattr(b, "metrics", None) or {}

        return {
            "entity_type": "trend",
            "comparison": [
                {"field": "title", "a": getattr(a, "title", ""), "b": getattr(b, "title", "")},
                {"field": "type", "a": str(getattr(a, "trend_type", "")), "b": str(getattr(b, "trend_type", ""))},
                {"field": "direction", "a": str(getattr(a, "trend_direction", "")), "b": str(getattr(b, "trend_direction", ""))},
                {"field": "score", "a": getattr(a_m, "trend_score", 0), "b": getattr(b_m, "trend_score", 0), "delta": getattr(b_m, "trend_score", 0) - getattr(a_m, "trend_score", 0)},
                {"field": "growth_pct", "a": getattr(a_m, "growth_pct", 0), "b": getattr(b_m, "growth_pct", 0)},
                {"field": "confidence", "a": getattr(a_m, "confidence", 0), "b": getattr(b_m, "confidence", 0)},
                {"field": "momentum", "a": getattr(a_m, "momentum", 0), "b": getattr(b_m, "momentum", 0)},
                {"field": "companies", "a": getattr(a, "affected_companies", []), "b": getattr(b, "affected_companies", [])},
            ],
            "a_id": a_id,
            "b_id": b_id,
        }

    def _compare_companies(self, a_name: str, b_name: str) -> dict[str, Any]:
        from phase3.opportunity.store import OpportunityStore
        store = OpportunityStore(self._config.phase3_dir)
        opps = store.load_opportunities()

        a_opps = [o for o in opps if a_name.lower() in [c.lower() for c in getattr(o, "affected_companies", [])]]
        b_opps = [o for o in opps if b_name.lower() in [c.lower() for c in getattr(o, "affected_companies", [])]]

        return {
            "entity_type": "company",
            "a": a_name,
            "b": b_name,
            "a_opportunity_count": len(a_opps),
            "b_opportunity_count": len(b_opps),
            "a_avg_score": sum(getattr(o, "opportunity_score", 0) for o in a_opps) / len(a_opps) if a_opps else 0,
            "b_avg_score": sum(getattr(o, "opportunity_score", 0) for o in b_opps) / len(b_opps) if b_opps else 0,
            "a_top_opportunities": [getattr(o, "title", "") for o in sorted(a_opps, key=lambda o: getattr(o, "opportunity_score", 0), reverse=True)[:3]],
            "b_top_opportunities": [getattr(o, "title", "") for o in sorted(b_opps, key=lambda o: getattr(o, "opportunity_score", 0), reverse=True)[:3]],
            "shared_technologies": list(set(
                t for o in a_opps for t in getattr(o, "affected_technologies", [])
            ) & set(
                t for o in b_opps for t in getattr(o, "affected_technologies", [])
            )),
        }

    def _compare_products(self, a_name: str, b_name: str) -> dict[str, Any]:
        from phase3.opportunity.store import OpportunityStore
        store = OpportunityStore(self._config.phase3_dir)
        opps = store.load_opportunities()

        a_opps = [o for o in opps if a_name.lower() in [p.lower() for p in getattr(o, "affected_products", [])]]
        b_opps = [o for o in opps if b_name.lower() in [p.lower() for p in getattr(o, "affected_products", [])]]

        return {
            "entity_type": "product",
            "a": a_name,
            "b": b_name,
            "a_opportunity_count": len(a_opps),
            "b_opportunity_count": len(b_opps),
            "a_avg_score": sum(getattr(o, "opportunity_score", 0) for o in a_opps) / len(a_opps) if a_opps else 0,
            "b_avg_score": sum(getattr(o, "opportunity_score", 0) for o in b_opps) / len(b_opps) if b_opps else 0,
        }

    def _find_opp(self, opps: list, opp_id: str) -> Any:
        for o in opps:
            if getattr(o, "opportunity_id", "") == opp_id:
                return o
            if getattr(o, "title", "").lower() == opp_id.lower():
                return o
        return None

    def _find_trend(self, trends: list, trend_id: str) -> Any:
        for t in trends:
            if getattr(t, "trend_id", "") == trend_id:
                return t
            if getattr(t, "title", "").lower() == trend_id.lower():
                return t
        return None

    def _build_citations(self, data: dict) -> list[Citation]:
        citations: list[Citation] = []
        for item in data.get("comparison", []):
            a_id = data.get("a_id", "")
            b_id = data.get("b_id", "")
            if a_id:
                citations.append(Citation(
                    source_module=CitationSource.OPPORTUNITY if data.get("entity_type") == "opportunity" else CitationSource.TREND,
                    source_id=a_id,
                    source_title=str(item.get("a", "")),
                    snippet=f"Field: {item.get('field', '')}",
                ))
            if b_id:
                citations.append(Citation(
                    source_module=CitationSource.OPPORTUNITY if data.get("entity_type") == "opportunity" else CitationSource.TREND,
                    source_id=b_id,
                    source_title=str(item.get("b", "")),
                    snippet=f"Field: {item.get('field', '')}",
                ))
        return citations[:8]
