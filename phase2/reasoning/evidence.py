"""Evidence aggregation — combine multiple evidence paths into conclusions."""

from __future__ import annotations

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.schema import EdgeType
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.schema import EvidenceAggregation


class EvidenceAggregator:
    def __init__(
        self,
        min_evidence_count: int = 2,
        conflicting_threshold: float = 0.3,
    ) -> None:
        self._min_evidence_count = min_evidence_count
        self._conflicting_threshold = conflicting_threshold

    def aggregate(
        self,
        graph: GraphInterface,
        conclusion_node_id: str,
        evidence_paths: list[list[str]],
        propagator: ConfidencePropagator,
    ) -> EvidenceAggregation:
        conclusion = graph.get_node(conclusion_node_id)
        conclusion_label = conclusion.label if conclusion else conclusion_node_id

        path_confidences = [
            propagator.propagate(path, graph) for path in evidence_paths
        ]
        path_confidences = [c for c in path_confidences if c > 0]

        evidence_node_ids: set[str] = set()
        for path in evidence_paths:
            for nid in path:
                if nid != conclusion_node_id:
                    evidence_node_ids.add(nid)

        conflicting = self.find_conflicting(graph, conclusion_node_id)
        aggregated_conf = 0.0
        if path_confidences:
            aggregated_conf = propagator.aggregate(path_confidences)

        return EvidenceAggregation(
            conclusion_node_id=conclusion_node_id,
            conclusion_label=conclusion_label,
            evidence_node_ids=sorted(evidence_node_ids),
            evidence_count=len(evidence_paths),
            aggregated_confidence=round(aggregated_conf, 4),
            aggregation_method=propagator.strategy.value,
            conflicting_evidence_count=len(conflicting),
        )

    def find_conflicting(
        self,
        graph: GraphInterface,
        node_id: str,
    ) -> list[str]:
        low_conf_nodes: list[str] = []
        for nid in [node_id]:
            incoming: list[str] = []
            try:
                raw = graph.predecessors(nid)
                incoming = [r for r in raw if isinstance(r, str)]
            except Exception:
                pass
            for src in incoming:
                src_node = graph.get_node(src)
                if src_node and src_node.confidence < self._conflicting_threshold:
                    low_conf_nodes.append(src)
        successors: list[str] = []
        try:
            raw = graph.successors(node_id)
            successors = [r for r in raw if isinstance(r, str)]
        except Exception:
            pass
        for succ in successors:
            succ_node = graph.get_node(succ)
            if succ_node and succ_node.confidence < self._conflicting_threshold:
                low_conf_nodes.append(succ)
        return low_conf_nodes

    def aggregate_all(
        self,
        graph: GraphInterface,
        propagator: ConfidencePropagator,
    ) -> list[EvidenceAggregation]:
        aggregations: list[EvidenceAggregation] = []
        all_nodes = [n.node_id for n in graph.nodes()]
        for nid in all_nodes:
            evidence_paths = self._collect_evidence_paths(graph, nid)
            if len(evidence_paths) >= self._min_evidence_count:
                agg = self.aggregate(graph, nid, evidence_paths, propagator)
                aggregations.append(agg)
        return aggregations

    def _collect_evidence_paths(
        self,
        graph: GraphInterface,
        node_id: str,
    ) -> list[list[str]]:
        paths: list[list[str]] = []
        evidence_types = [EdgeType.REFERENCES, EdgeType.SUPPORTED_BY, EdgeType.DERIVED_FROM]
        seen_sources: set[str] = set()

        for et in evidence_types:
            for src in graph.predecessors(node_id, edge_type=et):
                if src not in seen_sources:
                    seen_sources.add(src)
                    paths.append([src, node_id])
        return paths
