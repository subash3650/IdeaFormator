"""Evidence convergence rule — aggregate multiple evidence paths to a conclusion."""

from __future__ import annotations

import hashlib

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.schema import EdgeType
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.rules.base import ReasoningRule
from phase2.reasoning.rules.registry import register_rule
from phase2.reasoning.schema import InferenceResult, InferenceType


@register_rule(
    name="evidence_convergence",
    version="1.0",
    priority=90,
    dependencies=["transitive_closure"],
    description="Aggregate converging evidence: multiple paths to the same node increase confidence",
    author="system",
)
class EvidenceConvergenceRule(ReasoningRule):
    def __init__(self) -> None:
        self._min_evidence_count: int = 2

    @property
    def name(self) -> str:
        return "evidence_convergence"

    @property
    def description(self) -> str:
        return "Aggregate converging evidence: multiple paths to the same node increase confidence"

    def matches(self, graph: GraphInterface, node_id: str) -> bool:
        incoming_edges = graph.predecessors(node_id, edge_type=EdgeType.REFERENCES)
        incoming_supported = graph.predecessors(node_id, edge_type=EdgeType.SUPPORTED_BY)
        incoming_derived = graph.predecessors(node_id, edge_type=EdgeType.DERIVED_FROM)
        total = len(incoming_edges) + len(incoming_supported) + len(incoming_derived)
        return total >= self._min_evidence_count

    def apply(
        self,
        graph: GraphInterface,
        node_id: str,
        propagator: ConfidencePropagator,
    ) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        evidence_types = [EdgeType.REFERENCES, EdgeType.SUPPORTED_BY, EdgeType.DERIVED_FROM]
        all_sources: list[str] = []
        for et in evidence_types:
            all_sources.extend(graph.predecessors(node_id, edge_type=et))
        all_sources = list(dict.fromkeys(all_sources))

        if len(all_sources) < self._min_evidence_count:
            return results

        confidences: list[float] = []
        for src in all_sources:
            src_node = graph.get_node(src)
            target_node = graph.get_node(node_id)
            src_conf = src_node.confidence if src_node else 0.5
            tgt_conf = target_node.confidence if target_node else 0.5
            confidences.append(propagator.aggregate([src_conf, tgt_conf]))
        weights = [1.0] * len(confidences)
        aggregated_conf = propagator.aggregate(confidences, weights)

        if not propagator.above_threshold(aggregated_conf):
            return results

        inference_id = hashlib.sha256(
            f"evidence_convergence:{node_id}:{':'.join(sorted(all_sources))}:1.0".encode()
        ).hexdigest()
        chain_id = hashlib.sha256(f"chain:{inference_id}:1.0".encode()).hexdigest()

        results.append(InferenceResult(
            inference_id=inference_id,
            inference_type=InferenceType.EVIDENCE_AGGREGATION,
            derived_node_id=node_id,
            confidence=round(aggregated_conf, 4),
            chain_id=chain_id,
            provenance=[node_id] + all_sources,
        ))
        return results
