"""Causal chain rule — build causal chains from PROBLEM_SIGNAL nodes."""

from __future__ import annotations

import hashlib

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.schema import EdgeType, NodeType
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.rules.base import ReasoningRule
from phase2.reasoning.rules.registry import register_rule
from phase2.reasoning.schema import InferenceResult, InferenceType


@register_rule(
    name="causal_chain",
    version="1.0",
    priority=80,
    dependencies=["transitive_closure"],
    description="Build causal chains by tracing CAUSES edges from PROBLEM_SIGNAL nodes",
    author="system",
)
class CausalChainRule(ReasoningRule):
    def __init__(self) -> None:
        self._max_depth: int = 8

    @property
    def name(self) -> str:
        return "causal_chain"

    @property
    def description(self) -> str:
        return "Build causal chains by tracing CAUSES edges from PROBLEM_SIGNAL nodes"

    def matches(self, graph: GraphInterface, node_id: str) -> bool:
        node = graph.get_node(node_id)
        if node is None:
            return False
        if node.node_type == NodeType.PROBLEM_SIGNAL:
            predecessors = graph.predecessors(node_id, edge_type=EdgeType.CAUSES)
            return len(predecessors) > 0
        return False

    def apply(
        self,
        graph: GraphInterface,
        node_id: str,
        propagator: ConfidencePropagator,
    ) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        chains = self._build_causal_chains(graph, node_id, propagator)
        for chain_path, conf in chains:
            if len(chain_path) < 2:
                continue
            root = chain_path[0]
            effect = chain_path[-1]
            inference_id = hashlib.sha256(
                f"causal:{root}:{effect}:{'_'.join(chain_path)}:1.0".encode()
            ).hexdigest()
            chain_id = hashlib.sha256(f"chain:{inference_id}:1.0".encode()).hexdigest()
            results.append(InferenceResult(
                inference_id=inference_id,
                inference_type=InferenceType.CAUSAL_CHAIN,
                derived_edge_id=f"causal:{root}:{effect}",
                confidence=round(conf, 4),
                chain_id=chain_id,
                provenance=chain_path,
            ))
        return results

    def _build_causal_chains(
        self,
        graph: GraphInterface,
        node_id: str,
        propagator: ConfidencePropagator,
    ) -> list[tuple[list[str], float]]:
        chains: list[tuple[list[str], float]] = []
        visited: set[str] = set()

        def dfs(current: str, path: list[str], depth: int) -> None:
            if depth > self._max_depth:
                return
            predecessors = graph.predecessors(current, edge_type=EdgeType.CAUSES)
            if not predecessors:
                if len(path) >= 2:
                    conf = propagator.propagate(path, graph)
                    chains.append((list(path), conf))
                return
            for pred in predecessors:
                if pred in visited:
                    continue
                visited.add(pred)
                path.append(pred)
                dfs(pred, path, depth + 1)
                path.pop()
                visited.discard(pred)

        visited.add(node_id)
        dfs(node_id, [node_id], 0)
        return chains
