from __future__ import annotations

from pathlib import Path
from typing import Any

from phase3.presentation.config import PresentationConfig


class DataCollector:
    def __init__(self, config: PresentationConfig) -> None:
        self._config = config
        self._phase2 = config.phase2_dir
        self._phase3 = config.phase3_dir

    def collect_all(self) -> dict[str, Any]:
        return {
            "kg": self.collect_kg(),
            "reasoning": self.collect_reasoning(),
            "opportunities": self.collect_opportunities(),
            "trends": self.collect_trends(),
        }

    def collect_kg(self) -> dict[str, Any]:
        try:
            from phase2.knowledge_graph.store import KnowledgeGraphStore

            store = KnowledgeGraphStore(self._phase2)
            nodes = store.load_nodes()
            edges = store.load_edges()
            metadata = store.load_metadata()

            stats = store.count() if hasattr(store, "count") else {}
            if isinstance(metadata, dict):
                stats = metadata
            elif metadata and hasattr(metadata, "model_dump"):
                stats = metadata.model_dump(mode="json")

            return {
                "available": True,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "stats": stats,
                "nodes": nodes,
                "edges": edges,
            }
        except Exception:
            return {"available": False, "node_count": 0, "edge_count": 0, "stats": {}, "nodes": [], "edges": []}

    def collect_reasoning(self) -> dict[str, Any]:
        try:
            from phase2.reasoning.store import ReasoningStore

            store = ReasoningStore(self._phase2)
            inferences = store.load_inferences()
            chains = store.load_chains()
            root_causes = store.load_root_causes()
            evidence = store.load_evidence_aggregations()
            metadata = store.load_metadata()

            stats = {}
            if metadata and hasattr(metadata, "model_dump"):
                stats = metadata.model_dump(mode="json")

            return {
                "available": True,
                "inference_count": len(inferences),
                "chain_count": len(chains),
                "root_cause_count": len(root_causes),
                "evidence_count": len(evidence),
                "stats": stats,
                "inferences": inferences,
                "chains": chains,
                "root_causes": root_causes,
                "evidence_aggregations": evidence,
            }
        except Exception:
            return {
                "available": False,
                "inference_count": 0,
                "chain_count": 0,
                "root_cause_count": 0,
                "evidence_count": 0,
                "stats": {},
                "inferences": [],
                "chains": [],
                "root_causes": [],
                "evidence_aggregations": [],
            }

    def collect_opportunities(self) -> dict[str, Any]:
        try:
            from phase3.opportunity.store import OpportunityStore

            store = OpportunityStore(self._phase3)
            opportunities = store.load_opportunities()
            metadata = store.load_metadata()

            strong_pursue = [o for o in opportunities if o.recommendation_type and o.recommendation_type.value == "strong_pursue"]
            worth_exploring = [o for o in opportunities if o.recommendation_type and o.recommendation_type.value == "worth_exploring"]

            scores = [o.opportunity_score for o in opportunities if hasattr(o, "opportunity_score")]

            return {
                "available": True,
                "total": len(opportunities),
                "strong_pursue_count": len(strong_pursue),
                "worth_exploring_count": len(worth_exploring),
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "opportunities": opportunities,
                "top_opportunities": sorted(opportunities, key=lambda o: o.opportunity_score if hasattr(o, "opportunity_score") else 0, reverse=True)[:10],
            }
        except Exception:
            return {
                "available": False,
                "total": 0,
                "strong_pursue_count": 0,
                "worth_exploring_count": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "opportunities": [],
                "top_opportunities": [],
            }

    def collect_trends(self) -> dict[str, Any]:
        try:
            from phase3.trend.store import TrendStore

            store = TrendStore(self._phase3)
            trends = store.load_trends()
            metadata = store.load_metadata()

            growing = [t for t in trends if t.trend_type and t.trend_type.value == "growing"]
            declining = [t for t in trends if t.trend_type and t.trend_type.value == "declining"]
            emerging = [t for t in trends if t.trend_type and t.trend_type.value == "emerging"]

            scores = [t.metrics.trend_score for t in trends if hasattr(t, "metrics") and hasattr(t.metrics, "trend_score")]

            return {
                "available": True,
                "total": len(trends),
                "growing_count": len(growing),
                "declining_count": len(declining),
                "emerging_count": len(emerging),
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "trends": trends,
                "top_trends": sorted(trends, key=lambda t: t.metrics.trend_score if hasattr(t, "metrics") and hasattr(t.metrics, "trend_score") else 0, reverse=True)[:10],
            }
        except Exception:
            return {
                "available": False,
                "total": 0,
                "growing_count": 0,
                "declining_count": 0,
                "emerging_count": 0,
                "avg_score": 0.0,
                "trends": [],
                "top_trends": [],
            }
