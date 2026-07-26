"""Tests for reasoning rules and registry."""

from __future__ import annotations

import pytest

from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.rules import (
    CausalChainRule,
    EvidenceConvergenceRule,
    TransitiveClosureRule,
    available_rules,
    create_rule,
    get_rule_metadata,
    register_rule,
)
from phase2.reasoning.rules.base import ReasoningRule
from phase2.reasoning.schema import PropagationStrategy


class MockNode:
    def __init__(self, node_id: str, node_type=None, confidence: float = 0.5, label: str = ""):
        self.node_id = node_id
        self.node_type = node_type
        self.confidence = confidence
        self.label = label


class MockGraph:
    def __init__(self):
        self._nodes: dict[str, MockNode] = {}
        self._edges: list[tuple[str, str, str, float]] = []
        self._predecessors: dict[str, dict[str, list[str]]] = {}
        self._neighbors: dict[str, dict[str, list[str]]] = {}

    def add_node(self, node: MockNode) -> None:
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str):
        return self._nodes.get(node_id)

    def add_edge(self, src: str, tgt: str, etype: str, weight: float = 0.5) -> None:
        self._edges.append((src, tgt, etype, weight))
        self._neighbors.setdefault(src, {}).setdefault(etype, []).append(tgt)
        self._predecessors.setdefault(tgt, {}).setdefault(etype, []).append(src)

    def _resolve_etype(self, edge_type):
        if edge_type is not None:
            return edge_type.value if hasattr(edge_type, 'value') else edge_type
        return None

    def neighbors(self, node_id: str, edge_type=None, direction="out"):
        et = self._resolve_etype(edge_type)
        if et:
            return list(self._neighbors.get(node_id, {}).get(et, []))
        result: list[str] = []
        for lst in self._neighbors.get(node_id, {}).values():
            result.extend(lst)
        return result

    def predecessors(self, node_id: str, edge_type=None):
        et = self._resolve_etype(edge_type)
        if et:
            return list(self._predecessors.get(node_id, {}).get(et, []))
        result: list[str] = []
        for lst in self._predecessors.get(node_id, {}).values():
            result.extend(lst)
        return result

    def successors(self, node_id: str, edge_type=None):
        et = self._resolve_etype(edge_type)
        if et:
            return list(self._neighbors.get(node_id, {}).get(et, []))
        result: list[str] = []
        for lst in self._neighbors.get(node_id, {}).values():
            result.extend(lst)
        return result

    def get_edge(self, edge_id: str):
        for e in self._edges:
            if f"{e[2]}:{e[0]}:{e[1]}" == edge_id:
                return e
        return None

    def node_count(self):
        return len(self._nodes)

    def edge_count(self):
        return len(self._edges)

    def nodes(self):
        return list(self._nodes.values())

    def edges(self):
        return self._edges

    def nodes_by_type(self, ntype):
        return [n for n in self._nodes.values() if n.node_type == ntype]

    def out_degree(self, node_id: str) -> int:
        return len(self._neighbors.get(node_id, []))

    def in_degree(self, node_id: str) -> int:
        return len(self._predecessors.get(node_id, []))

    def degree(self, node_id: str) -> int:
        return self.out_degree(node_id) + self.in_degree(node_id)


class TestRuleRegistry:
    def test_available_rules(self) -> None:
        rules = available_rules()
        assert "transitive_closure" in rules
        assert "causal_chain" in rules
        assert "evidence_convergence" in rules

    def test_create_transitive(self) -> None:
        rule = create_rule("transitive_closure")
        assert isinstance(rule, TransitiveClosureRule)

    def test_create_causal(self) -> None:
        rule = create_rule("causal_chain")
        assert isinstance(rule, CausalChainRule)

    def test_create_evidence(self) -> None:
        rule = create_rule("evidence_convergence")
        assert isinstance(rule, EvidenceConvergenceRule)

    def test_get_metadata(self) -> None:
        meta = get_rule_metadata("transitive_closure")
        assert meta.name == "transitive_closure"
        assert meta.priority == 100
        assert meta.dependencies == []

    def test_get_metadata_causal(self) -> None:
        meta = get_rule_metadata("causal_chain")
        assert meta.priority == 80
        assert "transitive_closure" in meta.dependencies

    def test_get_metadata_evidence(self) -> None:
        meta = get_rule_metadata("evidence_convergence")
        assert meta.priority == 90
        assert "transitive_closure" in meta.dependencies


class TestTransitiveClosureRule:
    def test_applies_to_any_node(self) -> None:
        graph = MockGraph()
        graph.add_node(MockNode("n0"))
        rule = TransitiveClosureRule()
        assert rule.matches(graph, "n0") is True

    def test_derive_transitive_edge(self) -> None:
        from phase2.knowledge_graph.schema import EdgeType
        graph = MockGraph()
        for i in range(3):
            graph.add_node(MockNode(f"n{i}", confidence=0.8))
        graph.add_edge("n0", "n1", "causes")
        graph.add_edge("n1", "n2", "causes")
        rule = TransitiveClosureRule()
        rule.initialize(graph)
        propagator = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        results = rule.apply(graph, "n0", propagator)
        derived = [r for r in results if r.derived_edge_id == "causes:n0:n2"]
        assert len(derived) > 0
        assert derived[0].confidence > 0


class TestCausalChainRule:
    def test_matches_problem_signal_with_predecessors(self) -> None:
        from phase2.knowledge_graph.schema import NodeType, EdgeType
        graph = MockGraph()
        graph.add_node(MockNode("n0", node_type=NodeType.PROBLEM_SIGNAL))
        graph.add_node(MockNode("n1"))
        graph.add_edge("n1", "n0", "causes")
        rule = CausalChainRule()
        assert rule.matches(graph, "n0") is True

    def test_does_not_match_without_causes(self) -> None:
        from phase2.knowledge_graph.schema import NodeType
        graph = MockGraph()
        graph.add_node(MockNode("n0", node_type=NodeType.PROBLEM_SIGNAL))
        rule = CausalChainRule()
        assert rule.matches(graph, "n0") is False


class TestEvidenceConvergenceRule:
    def test_matches_with_multiple_references(self) -> None:
        from phase2.knowledge_graph.schema import NodeType
        graph = MockGraph()
        graph.add_node(MockNode("target"))
        for i in range(3):
            graph.add_node(MockNode(f"src{i}"))
            graph.add_edge(f"src{i}", "target", "references")
        rule = EvidenceConvergenceRule()
        assert rule.matches(graph, "target") is True

    def test_does_not_match_insufficient(self) -> None:
        graph = MockGraph()
        graph.add_node(MockNode("target"))
        graph.add_node(MockNode("src1"))
        graph.add_edge("src1", "target", "references")
        rule = EvidenceConvergenceRule()
        assert rule.matches(graph, "target") is False
