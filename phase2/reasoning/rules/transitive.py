"""Transitive closure rule — derived edges along transitive paths."""

from __future__ import annotations

import hashlib

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.schema import EdgeType
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.rules.base import ReasoningRule
from phase2.reasoning.rules.registry import register_rule
from phase2.reasoning.schema import InferenceResult, InferenceType


@register_rule(
    name="transitive_closure",
    version="1.0",
    priority=100,
    dependencies=[],
    description="Derive transitive edges: if A→B and B→C then A→C",
    author="system",
)
class TransitiveClosureRule(ReasoningRule):
    def __init__(self) -> None:
        self._transitive_types: set[str] = set()

    @property
    def name(self) -> str:
        return "transitive_closure"

    @property
    def description(self) -> str:
        return "Derive transitive edges: if A→B and B→C then A→C"

    def initialize(self, graph: GraphInterface) -> None:
        self._transitive_types = {
            EdgeType.CAUSES.value,
            EdgeType.DEPENDS_ON.value,
            EdgeType.CONTAINS.value,
            EdgeType.BELONGS_TO.value,
            EdgeType.DERIVED_FROM.value,
            EdgeType.REFERENCES.value,
        }

    def matches(self, graph: GraphInterface, node_id: str) -> bool:
        return True

    def apply(
        self,
        graph: GraphInterface,
        node_id: str,
        propagator: ConfidencePropagator,
    ) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        for edge_type_str in sorted(self._transitive_types):
            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                continue

            neighbors_out = graph.neighbors(node_id, edge_type=edge_type, direction="out")
            for neighbor in neighbors_out:
                neighbors_of_neighbor = graph.neighbors(neighbor, edge_type=edge_type, direction="out")
                for target in neighbors_of_neighbor:
                    if target == node_id:
                        continue
                    existing = graph.get_edge(f"{edge_type_str}:{node_id}:{target}")
                    if existing:
                        continue

                    node = graph.get_node(node_id)
                    neighbor_node = graph.get_node(neighbor)
                    target_node = graph.get_node(target)
                    node_conf = node.confidence if node else 0.5
                    neighbor_conf = neighbor_node.confidence if neighbor_node else 0.5
                    target_conf = target_node.confidence if target_node else 0.5
                    path_confs = [node_conf, neighbor_conf, target_conf]
                    path_conf = propagator.compute_path_confidence(path_confs)

                    if not propagator.above_threshold(path_conf):
                        continue

                    inference_id = hashlib.sha256(
                        f"transitive:{edge_type_str}:{node_id}:{neighbor}:{target}:1.0".encode()
                    ).hexdigest()
                    chain_id = hashlib.sha256(
                        f"chain:{inference_id}:1.0".encode()
                    ).hexdigest()

                    results.append(InferenceResult(
                        inference_id=inference_id,
                        inference_type=InferenceType.TRANSITIVE,
                        derived_edge_id=f"{edge_type_str}:{node_id}:{target}",
                        confidence=round(path_conf, 4),
                        chain_id=chain_id,
                        provenance=[node_id, neighbor, target],
                    ))
        return results
