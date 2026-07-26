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


@register_tool("evidence")
class EvidenceTool(BaseTool):
    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()

    @property
    def name(self) -> str:
        return "evidence"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Retrieve evidence supporting conclusions, opportunities, or trends",
            supported_intents=[
                Intent.EVIDENCE,
                Intent.EXPLAIN,
            ],
            priority=ToolPriority.HIGH,
            estimated_cost=3.0,
            estimated_latency_ms=400.0,
            dependencies=["reasoning", "kg"],
            permissions=PermissionType.READ_ONLY,
            cacheable=True,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        try:
            action = params.get("action", "for_conclusion")
            target_id = params.get("target_id", params.get("conclusion_id", ""))
            top_k = params.get("top_k", 10)

            if action == "for_opportunity":
                data = self._evidence_for_opportunity(target_id, top_k)
            elif action == "for_trend":
                data = self._evidence_for_trend(target_id, top_k)
            elif action == "for_conclusion":
                data = self._evidence_for_conclusion(target_id, top_k)
            else:
                data = self._evidence_for_conclusion(target_id, top_k)

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

    def _evidence_for_opportunity(self, opp_id: str, top_k: int) -> dict[str, Any]:
        from phase3.opportunity.store import OpportunityStore

        store = OpportunityStore(self._config.phase3_dir)
        opps = store.load_opportunities()
        target = None
        for o in opps:
            if getattr(o, "opportunity_id", "") == opp_id:
                target = o
                break

        if target is None:
            return {"error": f"Opportunity not found: {opp_id}"}

        evidence = getattr(target, "supporting_evidence", [])
        chain_ids = getattr(target, "reasoning_chain_ids", [])
        kg_ids = getattr(target, "kg_node_ids", [])

        result: dict[str, Any] = {
            "opportunity_id": opp_id,
            "title": getattr(target, "title", ""),
            "evidence_count": len(evidence),
            "reasoning_chain_count": len(chain_ids),
            "kg_node_count": len(kg_ids),
        }

        if evidence:
            result["supporting_evidence"] = evidence[:top_k]
        if chain_ids:
            result["reasoning_chain_ids"] = chain_ids

        from phase2.reasoning.store import ReasoningStore
        try:
            rs = ReasoningStore(self._config.phase2_dir)
            evidence_aggs = rs.load_evidence_aggregations()
            result["evidence_aggregations"] = [
                {
                    "conclusion_label": getattr(e, "conclusion_label", ""),
                    "evidence_count": getattr(e, "evidence_count", 0),
                    "aggregated_confidence": getattr(e, "aggregated_confidence", 0),
                }
                for e in evidence_aggs[:top_k]
            ]
        except Exception:
            pass

        return result

    def _evidence_for_trend(self, trend_id: str, top_k: int) -> dict[str, Any]:
        from phase3.trend.store import TrendStore
        from phase3.opportunity.store import OpportunityStore

        tstore = TrendStore(self._config.phase3_dir)
        trends = tstore.load_trends()
        target = None
        for t in trends:
            if getattr(t, "trend_id", "") == trend_id:
                target = t
                break

        if target is None:
            return {"error": f"Trend not found: {trend_id}"}

        metrics = getattr(target, "metrics", None) or {}
        result: dict[str, Any] = {
            "trend_id": trend_id,
            "title": getattr(target, "title", ""),
            "trend_type": str(getattr(target, "trend_type", "")),
            "summary": getattr(target, "summary", ""),
            "metrics": {
                "growth_pct": getattr(metrics, "growth_pct", 0),
                "confidence": getattr(metrics, "confidence", 0),
                "trend_score": getattr(metrics, "trend_score", 0),
                "observation_count": getattr(metrics, "total_observations", 0),
            },
            "snapshot_count": len(getattr(target, "snapshot_ids", [])),
            "correlation_count": len(getattr(target, "correlations", [])),
        }

        try:
            ostore = OpportunityStore(self._config.phase3_dir)
            opps = ostore.load_opportunities()
            related = [
                {
                    "title": getattr(o, "title", ""),
                    "score": getattr(o, "opportunity_score", 0),
                    "id": getattr(o, "opportunity_id", ""),
                }
                for o in opps
                if trend_id in getattr(o, "kg_node_ids", [])
            ]
            result["related_opportunities"] = related[:top_k]
        except Exception:
            pass

        return result

    def _evidence_for_conclusion(self, conclusion_id: str, top_k: int) -> dict[str, Any]:
        from phase2.reasoning.store import ReasoningStore
        from phase2.knowledge_graph.store import KnowledgeGraphStore

        rstore = ReasoningStore(self._config.phase2_dir)
        evidence = rstore.load_evidence_aggregations()
        inferences = rstore.load_inferences()

        result: dict[str, Any] = {
            "conclusion_id": conclusion_id,
            "evidence_aggregations": [],
            "inferences": [],
        }

        for e in evidence:
            if getattr(e, "conclusion_node_id", "") == conclusion_id:
                result["evidence_aggregations"].append({
                    "conclusion_label": getattr(e, "conclusion_label", ""),
                    "evidence_count": getattr(e, "evidence_count", 0),
                    "aggregated_confidence": getattr(e, "aggregated_confidence", 0),
                })

        for inf in inferences:
            if conclusion_id in getattr(inf, "provenance", []):
                result["inferences"].append({
                    "inference_id": getattr(inf, "inference_id", ""),
                    "inference_type": str(getattr(inf, "inference_type", "")),
                    "confidence": getattr(inf, "confidence", 0),
                })

        try:
            kg_store = KnowledgeGraphStore(self._config.phase2_dir)
            nodes = kg_store.load_nodes()
            for n in nodes:
                if getattr(n, "node_id", "") == conclusion_id:
                    result["node_label"] = getattr(n, "label", "")
                    result["node_type"] = str(getattr(n, "node_type", ""))
                    break
        except Exception:
            pass

        return result

    def _build_citations(self, data: dict) -> list[Citation]:
        citations: list[Citation] = []
        for ea in data.get("evidence_aggregations", []):
            citations.append(Citation(
                source_module=CitationSource.REASONING,
                source_id=data.get("conclusion_id", data.get("opportunity_id", data.get("trend_id", ""))),
                source_title=ea.get("conclusion_label", ""),
                confidence=ea.get("aggregated_confidence", 0),
                snippet=f"{ea.get('evidence_count', 0)} evidence items",
            ))
        for inf in data.get("inferences", []):
            citations.append(Citation(
                source_module=CitationSource.REASONING,
                source_id=inf.get("inference_id", ""),
                source_title=inf.get("inference_type", ""),
                confidence=inf.get("confidence", 0),
            ))
        return citations[:8]
