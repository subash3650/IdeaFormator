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


@register_tool("search")
class SearchTool(BaseTool):
    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()

    @property
    def name(self) -> str:
        return "search"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Search across all intelligence modules (KG, reasoning, opportunities, trends)",
            supported_intents=[
                Intent.SEARCH,
                Intent.UNKNOWN,
            ],
            priority=ToolPriority.MEDIUM,
            estimated_cost=4.0,
            estimated_latency_ms=500.0,
            permissions=PermissionType.READ_ONLY,
            cacheable=False,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        try:
            query = params.get("query", "")
            top_k = params.get("top_k", 5)
            modules = params.get("modules", ["kg", "opportunity", "trend"])

            results: dict[str, Any] = {"query": query}
            all_citations: list[Citation] = []

            if "kg" in modules:
                kg_data, kg_citations = self._search_kg(query, top_k)
                results["knowledge_graph"] = kg_data
                all_citations.extend(kg_citations)

            if "opportunity" in modules:
                opp_data, opp_citations = self._search_opportunities(query, top_k)
                results["opportunities"] = opp_data
                all_citations.extend(opp_citations)

            if "trend" in modules:
                trend_data, trend_citations = self._search_trends(query, top_k)
                results["trends"] = trend_data
                all_citations.extend(trend_citations)

            if "reasoning" in modules:
                reasoning_data, reasoning_citations = self._search_reasoning(query, top_k)
                results["reasoning"] = reasoning_data
                all_citations.extend(reasoning_citations)

            results["total_matches"] = sum(
                r.get("total", 0) for k, r in results.items()
                if isinstance(r, dict) and k != "query"
            )

            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=results,
                citations=all_citations,
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

    def _search_kg(self, query: str, top_k: int) -> tuple[dict, list[Citation]]:
        from phase2.knowledge_graph.store import KnowledgeGraphStore
        try:
            store = KnowledgeGraphStore(self._config.phase2_dir)
            nodes = store.load_nodes()
            q = query.lower()
            scored = []
            for n in nodes:
                label = (getattr(n, "label", "") or "").lower()
                nid = (getattr(n, "node_id", "") or "").lower()
                score = 0
                if q in label:
                    score += 2
                if q in nid:
                    score += 1
                if score > 0:
                    scored.append((score, n))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]
            citations = [
                Citation(
                    source_module=CitationSource.KNOWLEDGE_GRAPH,
                    source_id=getattr(n, "node_id", ""),
                    source_title=getattr(n, "label", ""),
                    confidence=getattr(n, "confidence", 0.0),
                    snippet=f"{getattr(n, 'node_type', '')}: {getattr(n, 'label', '')}",
                )
                for _, n in top
            ]
            return {"total": len(scored), "results": [getattr(n, "label", "") for _, n in top]}, citations
        except Exception:
            return {"total": 0, "results": []}, []

    def _search_opportunities(self, query: str, top_k: int) -> tuple[dict, list[Citation]]:
        from phase3.opportunity.store import OpportunityStore
        try:
            store = OpportunityStore(self._config.phase3_dir)
            opps = store.load_opportunities()
            q = query.lower()
            scored = []
            for o in opps:
                title = (getattr(o, "title", "") or "").lower()
                score = 3 if q in title else 0
                if score > 0:
                    scored.append((score, o))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]
            citations = [
                Citation(
                    source_module=CitationSource.OPPORTUNITY,
                    source_id=getattr(o, "opportunity_id", ""),
                    source_title=getattr(o, "title", ""),
                    confidence=getattr(o, "opportunity_score", 0) / 100.0,
                    snippet=f"Score: {getattr(o, 'opportunity_score', 0):.2f}",
                )
                for _, o in top
            ]
            return {"total": len(scored), "results": [getattr(o, "title", "") for _, o in top]}, citations
        except Exception:
            return {"total": 0, "results": []}, []

    def _search_trends(self, query: str, top_k: int) -> tuple[dict, list[Citation]]:
        from phase3.trend.store import TrendStore
        try:
            store = TrendStore(self._config.phase3_dir)
            trends = store.load_trends()
            q = query.lower()
            scored = []
            for t in trends:
                title = (getattr(t, "title", "") or "").lower()
                score = 3 if q in title else 0
                if score > 0:
                    scored.append((score, t))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]
            metrics = [getattr(t, "metrics", None) for _, t in top]
            citations = [
                Citation(
                    source_module=CitationSource.TREND,
                    source_id=getattr(t, "trend_id", ""),
                    source_title=getattr(t, "title", ""),
                    confidence=m.trend_score if m else 0,
                    snippet=f"Type: {getattr(t, 'trend_type', '')}",
                )
                for (_, t), m in zip(top, metrics)
            ]
            return {"total": len(scored), "results": [getattr(t, "title", "") for _, t in top]}, citations
        except Exception:
            return {"total": 0, "results": []}, []

    def _search_reasoning(self, query: str, top_k: int) -> tuple[dict, list[Citation]]:
        from phase2.reasoning.store import ReasoningStore
        try:
            store = ReasoningStore(self._config.phase2_dir)
            causes = store.load_root_causes()
            q = query.lower()
            matched = []
            for rc in causes:
                label = (getattr(rc, "cause_label", "") or "").lower()
                if q in label:
                    matched.append(rc)
            top = matched[:top_k]
            citations = [
                Citation(
                    source_module=CitationSource.REASONING,
                    source_id=getattr(rc, "cause_node_id", ""),
                    source_title=getattr(rc, "cause_label", ""),
                    confidence=getattr(rc, "propagated_confidence", 0),
                    snippet=f"Root cause for: {getattr(rc, 'effect_label', '')}",
                )
                for rc in top
            ]
            return {"total": len(matched), "results": [getattr(rc, "cause_label", "") for rc in top]}, citations
        except Exception:
            return {"total": 0, "results": []}, []
