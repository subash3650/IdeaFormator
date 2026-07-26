"""GraphEvaluator — health scores and metrics for knowledge graph."""

from __future__ import annotations

from typing import Any, TypedDict

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.schema import GraphMetadata


class StageHealth(TypedDict):
    score: float
    status: str


class GraphEvaluation(TypedDict):
    node_count: int
    edge_count: int
    density: float
    connected_components: int
    largest_component_ratio: float
    avg_confidence: float
    avg_degree: float
    orphan_node_count: int
    orphan_edge_count: int
    type_distribution: dict[str, int]
    edge_type_distribution: dict[str, int]
    health: StageHealth
    warnings: list[str]
    recommendations: list[str]


class GraphEvaluator:
    """Evaluates knowledge graph health and produces metrics."""

    def __init__(self, min_nodes: int = 10, min_edges: int = 5,
                 min_component_ratio: float = 0.5, min_avg_confidence: float = 0.6,
                 min_avg_degree: float = 1.0) -> None:
        self._min_nodes = min_nodes
        self._min_edges = min_edges
        self._min_component_ratio = min_component_ratio
        self._min_avg_confidence = min_avg_confidence
        self._min_avg_degree = min_avg_degree

    def evaluate(self, graph: GraphInterface) -> GraphEvaluation:
        all_nodes = graph.nodes()
        all_edges = graph.edges()
        node_count = len(all_nodes)
        edge_count = len(all_edges)

        type_dist: dict[str, int] = {}
        for n in all_nodes:
            type_dist[n.node_type.value] = type_dist.get(n.node_type.value, 0) + 1

        edge_type_dist: dict[str, int] = {}
        for e in all_edges:
            edge_type_dist[e.edge_type.value] = edge_type_dist.get(e.edge_type.value, 0) + 1

        density = (2 * edge_count) / (node_count * max(node_count - 1, 1)) if node_count > 1 else 0.0
        avg_conf = sum(n.confidence for n in all_nodes) / max(node_count, 1)
        avg_deg = sum(graph.degree(n.node_id) for n in all_nodes) / max(node_count, 1)
        orphan_nodes = sum(1 for n in all_nodes if graph.degree(n.node_id) == 0)
        orphan_edges = sum(1 for e in all_edges if graph.degree(e.source_node_id) == 0 or graph.degree(e.target_node_id) == 0)

        from phase2.knowledge_graph.algorithms import connected_components
        comps = connected_components(graph)
        cc_count = len(comps)
        largest_ratio = len(comps[0]) / max(node_count, 1) if comps else 0.0

        warnings: list[str] = []
        recommendations: list[str] = []

        if node_count < self._min_nodes:
            warnings.append(f"Node count ({node_count}) below minimum ({self._min_nodes})")
            recommendations.append("Increase the number of input observations or signals")
        if edge_count < self._min_edges:
            warnings.append(f"Edge count ({edge_count}) below minimum ({self._min_edges})")
            recommendations.append("Lower the minimum similarity threshold to generate more edges")
        if largest_ratio < self._min_component_ratio:
            warnings.append(f"Largest component ratio ({largest_ratio:.3f}) below minimum ({self._min_component_ratio})")
            recommendations.append("Consider reducing minimum confidence to connect more components")
        if avg_conf < self._min_avg_confidence:
            warnings.append(f"Average confidence ({avg_conf:.3f}) below minimum ({self._min_avg_confidence})")
            recommendations.append("Review upstream extraction quality to improve confidence scores")
        if avg_deg < self._min_avg_degree:
            warnings.append(f"Average degree ({avg_deg:.3f}) below minimum ({self._min_avg_degree})")
            recommendations.append("Nodes are sparsely connected; generate more relationships")

        # Composite health score (0-100)
        score = 100.0
        score -= max(0, (self._min_nodes - node_count) / self._min_nodes * 20) if self._min_nodes > 0 else 0
        score -= max(0, (self._min_edges - edge_count) / self._min_edges * 20) if self._min_edges > 0 else 0
        score -= max(0, (self._min_component_ratio - largest_ratio) / self._min_component_ratio * 20) if self._min_component_ratio > 0 else 0
        score -= max(0, (self._min_avg_confidence - avg_conf) / self._min_avg_confidence * 20) if self._min_avg_confidence > 0 else 0
        score -= max(0, (self._min_avg_degree - avg_deg) / self._min_avg_degree * 20) if self._min_avg_degree > 0 else 0
        score = max(0.0, min(100.0, score))

        status = "healthy" if score >= 80 else "degraded" if score >= 50 else "critical"

        return GraphEvaluation(
            node_count=node_count,
            edge_count=edge_count,
            density=round(density, 6),
            connected_components=cc_count,
            largest_component_ratio=round(largest_ratio, 6),
            avg_confidence=round(avg_conf, 6),
            avg_degree=round(avg_deg, 6),
            orphan_node_count=orphan_nodes,
            orphan_edge_count=orphan_edges,
            type_distribution=type_dist,
            edge_type_distribution=edge_type_dist,
            health=StageHealth(score=round(score, 2), status=status),
            warnings=warnings,
            recommendations=recommendations,
        )

    def evaluate_metadata(self, metadata: GraphMetadata) -> StageHealth:
        score = 100.0
        if metadata.node_count < self._min_nodes:
            score -= 20
        if metadata.edge_count < self._min_edges:
            score -= 20
        if metadata.largest_component_size > 0 and metadata.node_count > 0:
            ratio = metadata.largest_component_size / metadata.node_count
            if ratio < self._min_component_ratio:
                score -= 20
        if metadata.avg_confidence < self._min_avg_confidence:
            score -= 20
        if metadata.avg_degree < self._min_avg_degree:
            score -= 20
        score = max(0.0, min(100.0, score))
        status = "healthy" if score >= 80 else "degraded" if score >= 50 else "critical"
        return StageHealth(score=round(score, 2), status=status)
