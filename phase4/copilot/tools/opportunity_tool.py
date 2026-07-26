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


@register_tool("opportunity")
class OpportunityTool(BaseTool):
    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()

    @property
    def name(self) -> str:
        return "opportunity"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Query discovered business opportunities with scoring and recommendations",
            supported_intents=[
                Intent.QUERY_OPPORTUNITY,
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
            opportunities = store.load_opportunities()

            if action == "stats":
                data = self._stats(opportunities)
            elif action == "top":
                data = self._top(opportunities, params)
            elif action == "detail":
                data = self._detail(opportunities, params)
            elif action == "by_company":
                data = self._by_company(opportunities, params)
            elif action == "by_product":
                data = self._by_product(opportunities, params)
            elif action == "by_type":
                data = self._by_type(opportunities, params)
            else:
                data = self._search(opportunities, params)

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
        from phase3.opportunity.store import OpportunityStore
        return OpportunityStore(self._config.phase3_dir)

    def _stats(self, opps: list) -> dict[str, Any]:
        scores = [getattr(o, "opportunity_score", 0) for o in opps if hasattr(o, "opportunity_score")]
        recs: dict[str, int] = {}
        for o in opps:
            rt = str(getattr(o, "recommendation_type", ""))
            recs[rt] = recs.get(rt, 0) + 1
        return {
            "total": len(opps),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "recommendation_distribution": recs,
        }

    def _top(self, opps: list, params: dict) -> dict[str, Any]:
        limit = params.get("top_k", 10)
        sorted_opps = sorted(
            opps,
            key=lambda o: getattr(o, "opportunity_score", 0),
            reverse=True,
        )
        return {
            "results": [self._serialize(o) for o in sorted_opps[:limit]],
            "total": len(opps),
        }

    def _detail(self, opps: list, params: dict) -> dict[str, Any]:
        opp_id = params.get("opportunity_id", "")
        for o in opps:
            if getattr(o, "opportunity_id", "") == opp_id:
                return self._serialize(o, detailed=True)
        return {"error": f"Opportunity not found: {opp_id}"}

    def _by_company(self, opps: list, params: dict) -> dict[str, Any]:
        company = (params.get("company", "") or "").lower()
        results = []
        for o in opps:
            companies = [str(c).lower() for c in getattr(o, "affected_companies", [])]
            if company in companies:
                results.append(self._serialize(o))
        return {"company": company, "total": len(results), "results": results}

    def _by_product(self, opps: list, params: dict) -> dict[str, Any]:
        product = (params.get("product", "") or "").lower()
        results = []
        for o in opps:
            products = [str(p).lower() for p in getattr(o, "affected_products", [])]
            if product in products:
                results.append(self._serialize(o))
        return {"product": product, "total": len(results), "results": results}

    def _by_type(self, opps: list, params: dict) -> dict[str, Any]:
        opp_type = (params.get("opportunity_type", "") or "").lower()
        results = []
        for o in opps:
            if str(getattr(o, "suggested_business_model", "")).lower() == opp_type:
                results.append(self._serialize(o))
        return {"business_model": opp_type, "total": len(results), "results": results}

    def _search(self, opps: list, params: dict) -> dict[str, Any]:
        query = (params.get("query", "") or "").lower()
        limit = params.get("top_k", 10)
        scored = []
        for o in opps:
            title = (getattr(o, "title", "") or "").lower()
            summary = (getattr(o, "summary", "") or "").lower()
            problem = (getattr(o, "root_problem", "") or "").lower()
            companies = " ".join(getattr(o, "affected_companies", [])).lower()
            technologies = " ".join(getattr(o, "affected_technologies", [])).lower()

            score = 0
            if query in title:
                score += 3
            if query in summary:
                score += 2
            if query in problem:
                score += 2
            if query in companies or query in technologies:
                score += 1

            if score > 0:
                scored.append((score, self._serialize(o)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return {
            "query": params.get("query", ""),
            "total": len(scored),
            "results": [s[1] for s in scored[:limit]],
        }

    def _serialize(self, o: Any, detailed: bool = False) -> dict[str, Any]:
        result = {
            "opportunity_id": getattr(o, "opportunity_id", ""),
            "title": getattr(o, "title", ""),
            "opportunity_score": getattr(o, "opportunity_score", 0),
            "recommendation_type": str(getattr(o, "recommendation_type", "")),
            "business_model": str(getattr(o, "suggested_business_model", "")),
            "companies": getattr(o, "affected_companies", []),
            "technologies": getattr(o, "affected_technologies", []),
            "products": getattr(o, "affected_products", []),
        }
        if detailed:
            result.update({
                "summary": getattr(o, "summary", ""),
                "root_problem": getattr(o, "root_problem", ""),
                "suggested_solution": getattr(o, "suggested_solution", ""),
                "pain_severity": getattr(o, "pain_severity", 0),
                "trend_score": getattr(o, "trend_score", 0),
                "market_size": str(getattr(o, "estimated_market_size", "")),
                "confidence": str(getattr(o, "confidence", "")),
                "evidence_count": len(getattr(o, "supporting_evidence", [])),
            })
        return result

    def _build_citations(self, data: dict) -> list[Citation]:
        citations: list[Citation] = []
        for result in data.get("results", []):
            oid = result.get("opportunity_id", "")
            if oid:
                citations.append(Citation(
                    source_module=CitationSource.OPPORTUNITY,
                    source_id=oid,
                    source_title=result.get("title", ""),
                    confidence=result.get("opportunity_score", 0) / 100.0,
                    snippet=f"Score: {result.get('opportunity_score', 0):.2f}, Rec: {result.get('recommendation_type', '')}",
                ))
        return citations
