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


@register_tool("reasoning")
class ReasoningTool(BaseTool):
    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()

    @property
    def name(self) -> str:
        return "reasoning"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Query reasoning chains, inferences, root causes, and explanations",
            supported_intents=[
                Intent.QUERY_REASONING,
                Intent.EXPLAIN,
                Intent.EVIDENCE,
                Intent.STATISTICS,
            ],
            priority=ToolPriority.HIGH,
            estimated_cost=3.0,
            estimated_latency_ms=300.0,
            permissions=PermissionType.READ_ONLY,
            cacheable=True,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        try:
            action = params.get("action", "stats")
            store = self._get_store()

            if action == "stats":
                data = self._stats(store)
            elif action == "inferences":
                data = self._inferences(store, params)
            elif action == "chains":
                data = self._chains(store, params)
            elif action == "root_causes":
                data = self._root_causes(store, params)
            elif action == "explain":
                data = self._explain(store, params)
            elif action == "evidence":
                data = self._evidence(store, params)
            else:
                data = self._stats(store)

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
        from phase2.reasoning.store import ReasoningStore
        return ReasoningStore(self._config.phase2_dir)

    def _stats(self, store: Any) -> dict[str, Any]:
        inferences = store.load_inferences()
        chains = store.load_chains()
        causes = store.load_root_causes()
        evidence = store.load_evidence_aggregations()
        return {
            "inference_count": len(inferences),
            "chain_count": len(chains),
            "root_cause_count": len(causes),
            "evidence_count": len(evidence),
        }

    def _inferences(self, store: Any, params: dict) -> dict[str, Any]:
        inferences = store.load_inferences()
        limit = params.get("top_k", 10)
        inference_type = params.get("inference_type")
        results = []
        for inf in inferences:
            if inference_type and str(getattr(inf, "inference_type", "")) != inference_type:
                continue
            results.append({
                "inference_id": getattr(inf, "inference_id", ""),
                "inference_type": str(getattr(inf, "inference_type", "")),
                "confidence": getattr(inf, "confidence", 0.0),
                "chain_id": getattr(inf, "chain_id", ""),
                "created_at": getattr(inf, "created_at", ""),
            })
        return {"total": len(results), "inferences": results[:limit]}

    def _chains(self, store: Any, params: dict) -> dict[str, Any]:
        chains = store.load_chains()
        limit = params.get("top_k", 10)
        inference_id = params.get("inference_id")
        results = []
        for c in chains:
            if inference_id and getattr(c, "inference_id", "") != inference_id:
                continue
            results.append({
                "chain_id": getattr(c, "chain_id", ""),
                "inference_id": getattr(c, "inference_id", ""),
                "steps": len(getattr(c, "steps", [])),
                "total_confidence": getattr(c, "total_confidence", 0.0),
            })
        return {"total": len(results), "chains": results[:limit]}

    def _root_causes(self, store: Any, params: dict) -> dict[str, Any]:
        causes = store.load_root_causes()
        limit = params.get("top_k", 10)
        effect_id = params.get("effect_id")
        results = []
        for rc in causes:
            if effect_id and getattr(rc, "effect_node_id", "") != effect_id:
                continue
            results.append({
                "cause_node_id": getattr(rc, "cause_node_id", ""),
                "cause_label": getattr(rc, "cause_label", ""),
                "effect_node_id": getattr(rc, "effect_node_id", ""),
                "effect_label": getattr(rc, "effect_label", ""),
                "ranking_score": getattr(rc, "ranking_score", 0.0),
                "propagated_confidence": getattr(rc, "propagated_confidence", 0.0),
                "evidence_count": getattr(rc, "evidence_count", 0),
            })
        return {"total": len(results), "root_causes": results[:limit]}

    def _explain(self, store: Any, params: dict) -> dict[str, Any]:
        from phase2.reasoning.explanation import ExplanationGenerator
        from phase2.reasoning.rule_engine import RuleEngine, InferenceEngine
        from phase2.knowledge_graph.store import KnowledgeGraphStore
        from phase2.knowledge_graph.graph import CustomGraph

        inference_id = params.get("inference_id", "")
        inferences = store.load_inferences()
        inf = None
        for i in inferences:
            if getattr(i, "inference_id", "") == inference_id:
                inf = i
                break

        if inf is None:
            return {"error": f"Inference not found: {inference_id}"}

        chains = store.load_chains()
        chain = None
        for c in chains:
            if getattr(c, "inference_id", "") == inference_id:
                chain = c
                break

        kg_store = KnowledgeGraphStore(self._config.phase2_dir)
        nodes = kg_store.load_nodes()
        edges = kg_store.load_edges()
        graph = CustomGraph()
        for n in nodes:
            graph.add_node(n)
        for e in edges:
            graph.add_edge(e)

        gen = ExplanationGenerator()
        explanation = gen.explain_inference(inf, chain or c, graph)
        return {
            "inference_id": inference_id,
            "title": getattr(explanation, "title", ""),
            "summary": getattr(explanation, "summary", ""),
            "steps": getattr(explanation, "steps", []),
            "confidence": getattr(inf, "confidence", 0.0),
        }

    def _evidence(self, store: Any, params: dict) -> dict[str, Any]:
        evidence = store.load_evidence_aggregations()
        limit = params.get("top_k", 10)
        conclusion_id = params.get("conclusion_id")
        results = []
        for e in evidence:
            if conclusion_id and getattr(e, "conclusion_node_id", "") != conclusion_id:
                continue
            results.append({
                "conclusion_node_id": getattr(e, "conclusion_node_id", ""),
                "conclusion_label": getattr(e, "conclusion_label", ""),
                "evidence_count": getattr(e, "evidence_count", 0),
                "aggregated_confidence": getattr(e, "aggregated_confidence", 0.0),
            })
        return {"total": len(results), "evidence": results[:limit]}

    def _build_citations(self, data: dict) -> list[Citation]:
        citations: list[Citation] = []
        for item in data.get("root_causes", []):
            citations.append(Citation(
                source_module=CitationSource.REASONING,
                source_id=item.get("cause_node_id", ""),
                source_title=item.get("cause_label", ""),
                confidence=item.get("propagated_confidence", 0.0),
                snippet=f"Root cause: {item.get('cause_label', '')}",
            ))
        for item in data.get("evidence", []):
            citations.append(Citation(
                source_module=CitationSource.REASONING,
                source_id=item.get("conclusion_node_id", ""),
                source_title=item.get("conclusion_label", ""),
                confidence=item.get("aggregated_confidence", 0.0),
                snippet=f"{item.get('evidence_count', 0)} evidence items",
            ))
        return citations
