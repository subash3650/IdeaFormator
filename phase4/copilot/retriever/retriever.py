from __future__ import annotations

from typing import Any

from phase4.copilot.config import CopilotConfig


class UnifiedRetriever:
    def __init__(self, config: CopilotConfig) -> None:
        self._config = config
        self._stores: dict[str, Any] = {}

    def retrieve_kg(self, query: str = "", node_type: str | None = None, node_id: str | None = None) -> dict[str, Any]:
        try:
            from phase2.knowledge_graph.store import KnowledgeGraphStore
            store = self._store("kg", KnowledgeGraphStore, self._config.phase2_dir)
            nodes = store.load_nodes()
            edges = store.load_edges()
            result: dict[str, Any] = {"available": True, "node_count": len(nodes), "edge_count": len(edges)}

            if query:
                q = query.lower()
                matched = [n for n in nodes if q in (getattr(n, "label", "") or "").lower()]
                result["matched"] = len(matched)
                result["results"] = [getattr(n, "label", "") for n in matched[:5]]

            if node_type:
                matched = [n for n in nodes if str(getattr(n, "node_type", "")).lower() == node_type.lower()]
                result["by_type_count"] = len(matched)

            if node_id:
                target = None
                for n in nodes:
                    if getattr(n, "node_id", "") == node_id:
                        target = n
                        break
                if target:
                    result["node"] = {
                        "id": getattr(target, "node_id", ""),
                        "type": str(getattr(target, "node_type", "")),
                        "label": getattr(target, "label", ""),
                        "confidence": getattr(target, "confidence", 0),
                    }
                    neighbor_ids = set()
                    for e in edges:
                        if getattr(e, "source_node_id", "") == node_id:
                            neighbor_ids.add(getattr(e, "target_node_id", ""))
                        if getattr(e, "target_node_id", "") == node_id:
                            neighbor_ids.add(getattr(e, "source_node_id", ""))
                    result["neighbors"] = len(neighbor_ids)

            return result
        except Exception as e:
            return {"available": False, "error": str(e)}

    def retrieve_reasoning(self) -> dict[str, Any]:
        try:
            from phase2.reasoning.store import ReasoningStore
            store = self._store("reasoning", ReasoningStore, self._config.phase2_dir)
            inferences = store.load_inferences()
            chains = store.load_chains()
            causes = store.load_root_causes()
            evidence = store.load_evidence_aggregations()
            return {
                "available": True,
                "inference_count": len(inferences),
                "chain_count": len(chains),
                "root_cause_count": len(causes),
                "evidence_count": len(evidence),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def retrieve_opportunities(self, query: str = "") -> dict[str, Any]:
        try:
            from phase3.opportunity.store import OpportunityStore
            store = self._store("opportunity", OpportunityStore, self._config.phase3_dir)
            opps = store.load_opportunities()
            result: dict[str, Any] = {"available": True, "total": len(opps)}

            scores = [getattr(o, "opportunity_score", 0) for o in opps]
            result["avg_score"] = sum(scores) / len(scores) if scores else 0.0
            result["max_score"] = max(scores) if scores else 0.0

            if query:
                q = query.lower()
                matched = [o for o in opps if q in (getattr(o, "title", "") or "").lower()]
                result["matched"] = len(matched)
                result["results"] = [getattr(o, "title", "") for o in matched[:5]]

            result["recommendation_distribution"] = self._count_field(opps, "recommendation_type")
            result["business_model_distribution"] = self._count_field(opps, "suggested_business_model")

            return result
        except Exception as e:
            return {"available": False, "error": str(e)}

    def retrieve_trends(self, query: str = "") -> dict[str, Any]:
        try:
            from phase3.trend.store import TrendStore
            store = self._store("trend", TrendStore, self._config.phase3_dir)
            trends = store.load_trends()
            result: dict[str, Any] = {"available": True, "total": len(trends)}

            type_dist: dict[str, int] = {}
            for t in trends:
                tt = str(getattr(t, "trend_type", ""))
                type_dist[tt] = type_dist.get(tt, 0) + 1
            result["type_distribution"] = type_dist

            if query:
                q = query.lower()
                matched = [t for t in trends if q in (getattr(t, "title", "") or "").lower()]
                result["matched"] = len(matched)
                result["results"] = [getattr(t, "title", "") for t in matched[:5]]

            return result
        except Exception as e:
            return {"available": False, "error": str(e)}

    def retrieve_evidence(self, conclusion_id: str) -> dict[str, Any]:
        try:
            from phase2.reasoning.store import ReasoningStore
            store = self._store("reasoning", ReasoningStore, self._config.phase2_dir)
            evidence = store.load_evidence_aggregations()
            matched = [e for e in evidence if getattr(e, "conclusion_node_id", "") == conclusion_id]
            return {
                "available": True,
                "count": len(matched),
                "items": [
                    {
                        "label": getattr(e, "conclusion_label", ""),
                        "evidence_count": getattr(e, "evidence_count", 0),
                        "confidence": getattr(e, "aggregated_confidence", 0),
                    }
                    for e in matched[:10]
                ],
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def _store(self, name: str, cls: type, *args: Any) -> Any:
        if name not in self._stores:
            self._stores[name] = cls(*args)
        return self._stores[name]

    def _count_field(self, items: list, field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in items:
            val = str(getattr(item, field, ""))
            result[val] = result.get(val, 0) + 1
        return result
