"""KnowledgeGraphEngine — main orchestrator for Phase 3 Module 1."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase2.knowledge_graph.builder import KnowledgeGraphBuilder
from phase2.knowledge_graph.config import KnowledgeGraphConfig
from phase2.knowledge_graph.evaluator import GraphEvaluator
from phase2.knowledge_graph.exporter import GraphExporter
from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.search import GraphSearch
from phase2.knowledge_graph.store import KnowledgeGraphStore
from phase2.knowledge_graph.validator import GraphValidator


class KnowledgeGraphEngine:
    """High-level orchestrator for the Knowledge Graph Infrastructure.

    Flow:
        Pipeline assets → KnowledgeGraphBuilder → CustomGraph
        → GraphValidator → GraphEvaluator → GraphSearch → GraphExporter
    """

    def __init__(self, config: KnowledgeGraphConfig, run_id: str | None = None) -> None:
        self._config = config
        self._store = KnowledgeGraphStore(config.output_dir)
        self._builder = KnowledgeGraphBuilder(config, run_id)
        self._evaluator = GraphEvaluator()
        self._validator = GraphValidator()
        self._graph: GraphInterface | None = None

    @property
    def store(self) -> KnowledgeGraphStore:
        return self._store

    @property
    def graph(self) -> GraphInterface | None:
        return self._graph

    def build(self, force: bool = False) -> dict[str, Any]:
        """Build the knowledge graph from pipeline assets.

        Returns build summary including node count, edge count, health score.
        """
        start = time.perf_counter()

        graph = self._builder.build(force=force)
        self._graph = graph

        validation = self._validator.validate(graph)
        evaluation = self._evaluator.evaluate(graph)
        meta = graph.metadata(self._builder.run_id)

        elapsed = time.perf_counter() - start

        # Save evaluation outputs
        self._store.save_report(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_id": self._builder.run_id,
                "evaluation": dict(evaluation["health"]),
                "node_count": evaluation["node_count"],
                "edge_count": evaluation["edge_count"],
                "density": evaluation["density"],
                "connected_components": evaluation["connected_components"],
                "largest_component_ratio": evaluation["largest_component_ratio"],
                "avg_confidence": evaluation["avg_confidence"],
                "avg_degree": evaluation["avg_degree"],
                "orphan_node_count": evaluation["orphan_node_count"],
                "orphan_edge_count": evaluation["orphan_edge_count"],
                "warnings": evaluation["warnings"],
                "recommendations": evaluation["recommendations"],
            }
        )
        self._store.save_statistics(
            {
                "node_count": meta.node_count,
                "edge_count": meta.edge_count,
                "node_type_counts": meta.node_type_counts,
                "edge_type_counts": meta.edge_type_counts,
                "density": meta.density,
                "avg_confidence": meta.avg_confidence,
                "avg_degree": meta.avg_degree,
                "orphan_node_count": meta.orphan_node_count,
                "connected_components": meta.connected_components,
                "largest_component_size": meta.largest_component_size,
                "degree_distribution": self._compute_degree_distribution(graph),
            }
        )

        # Dashboard
        exporter = GraphExporter(graph, self._store, self._evaluator)
        exporter.export_dashboard(self._store.dashboard_json_path)
        exporter.export_summary(self._store.summary_path)

        return {
            "node_count": evaluation["node_count"],
            "edge_count": evaluation["edge_count"],
            "density": evaluation["density"],
            "connected_components": evaluation["connected_components"],
            "largest_component_size": meta.largest_component_size,
            "health_score": evaluation["health"]["score"],
            "health_status": evaluation["health"]["status"],
            "valid": validation.valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "elapsed_seconds": round(elapsed, 4),
        }

    def search(self, query: str, top_k: int = 10, type_filter: str = "") -> list[dict[str, Any]]:
        """Search the knowledge graph by node ID, source ID, or label."""
        graph = self._load_or_rebuild()
        if graph is None:
            return []
        searcher = GraphSearch(graph)

        results: list[dict[str, Any]] = []

        # Try exact node ID first
        node = searcher.find_node(query)
        if node is not None:
            if not type_filter or node.node_type.value == type_filter:
                results.append(self._node_to_dict(node))

        # Try source_id lookup (in metadata)
        for n in graph.nodes():
            if n.metadata.get("source_id") == query or n.metadata.get("source_id") == query:
                if not type_filter or n.node_type.value == type_filter:
                    if n.node_id not in {r["node_id"] for r in results}:
                        results.append(self._node_to_dict(n))

        # Try label search
        label_results = searcher.find_by_label(query, fuzzy=True)
        for n in label_results:
            if not type_filter or n.node_type.value == type_filter:
                if n.node_id not in {r["node_id"] for r in results}:
                    results.append(self._node_to_dict(n))

        # Try type-specific search
        if not type_filter:
            results.extend(self._type_search(searcher, query))

        return sorted(results, key=lambda x: -x["confidence"])[:top_k]

    def stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        graph = self._load_or_rebuild()
        if graph is None:
            return {"node_count": 0, "edge_count": 0}

        evaluation = self._evaluator.evaluate(graph)
        return {
            "node_count": evaluation["node_count"],
            "edge_count": evaluation["edge_count"],
            "density": evaluation["density"],
            "connected_components": evaluation["connected_components"],
            "largest_component_ratio": evaluation["largest_component_ratio"],
            "avg_confidence": evaluation["avg_confidence"],
            "avg_degree": evaluation["avg_degree"],
            "orphan_node_count": evaluation["orphan_node_count"],
            "orphan_edge_count": evaluation["orphan_edge_count"],
            "health_score": evaluation["health"]["score"],
            "type_distribution": evaluation["type_distribution"],
            "edge_type_distribution": evaluation["edge_type_distribution"],
        }

    def verify(self) -> dict[str, Any]:
        """Verify knowledge graph integrity."""
        graph = self._load_or_rebuild()
        if graph is None:
            return {"valid": False, "node_count": 0, "edge_count": 0, "errors": ["No graph data found"], "warnings": []}

        validation = self._validator.validate(graph)
        return {
            "valid": validation.valid,
            "node_count": validation.node_count,
            "edge_count": validation.edge_count,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "duplicate_node_count": validation.duplicate_node_count,
            "duplicate_edge_count": validation.duplicate_edge_count,
            "orphan_edge_count": validation.orphan_edge_count,
            "self_loop_count": validation.self_loop_count,
            "cycle_count": validation.cycle_count,
            "schema_mismatch_count": validation.schema_mismatch_count,
        }

    def export(self, format: str = "all") -> dict[str, Any]:
        """Export knowledge graph in specified format."""
        graph = self._load_or_rebuild()
        if graph is None:
            return {"error": "No graph data found"}

        exporter = GraphExporter(graph, self._store, self._evaluator)
        graph_dir = self._store.get_graph_dir()
        result: dict[str, Any] = {}

        if format in ("all", "gexf"):
            result["gexf"] = str(exporter.export_gexf(graph_dir / "graph.gexf"))
        if format in ("all", "json"):
            result["json"] = str(exporter.export_json(graph_dir / "graph.json"))
        if format in ("all", "csv"):
            result["csv"] = {k: str(v) for k, v in exporter.export_csv(graph_dir).items()}
        if format in ("all", "report"):
            result["report"] = str(self._store.report_path)
            result["statistics"] = str(self._store.statistics_path)
            result["dashboard"] = str(self._store.dashboard_json_path)
            result["summary"] = str(self._store.summary_path)

        return result

    # ── Private ───────────────────────────────────────────────────────

    def _load_or_rebuild(self) -> GraphInterface | None:
        if self._graph is not None:
            return self._graph
        if self._store.exists():
            from phase2.knowledge_graph.graph import CustomGraph
            graph = CustomGraph()
            for node in self._store.load_nodes():
                graph.add_node(node)
            for edge in self._store.load_edges():
                graph.add_edge(edge)
            self._graph = graph
            return graph
        return None

    def _node_to_dict(self, node: Any) -> dict[str, Any]:
        return {
            "node_id": node.node_id,
            "node_type": node.node_type.value,
            "label": node.label,
            "confidence": node.confidence,
            "source_asset": node.source_asset,
        }

    def _type_search(self, searcher: GraphSearch, query: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for find_fn, type_name in [
            (lambda: searcher.find_observations(), "observation"),
            (lambda: searcher.find_evidence(), "evidence"),
            (lambda: searcher.find_problem_signals(), "problem_signal"),
            (lambda: searcher.find_clusters(), "cluster"),
        ]:
            try:
                for node in find_fn():
                    if query.lower() in node.label.lower():
                        results.append(self._node_to_dict(node))
            except Exception:
                continue
        return results

    def _compute_degree_distribution(self, graph: GraphInterface) -> dict[str, int]:
        dist: dict[str, int] = {}
        for node in graph.nodes():
            deg = graph.degree(node.node_id)
            dist[str(deg)] = dist.get(str(deg), 0) + 1
        return dist
