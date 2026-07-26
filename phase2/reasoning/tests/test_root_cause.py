"""Tests for RootCauseDiscoverer."""

from __future__ import annotations

import pytest

from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.root_cause import RootCauseDiscoverer
from phase2.reasoning.schema import PropagationStrategy, RootCauseRanking


class MockNode:
    def __init__(self, node_id: str, confidence=0.5, label="", node_type=None):
        self.node_id = node_id
        self.confidence = confidence
        self.label = label
        self.node_type = node_type


class MockGraph:
    def __init__(self):
        self._nodes: dict[str, MockNode] = {}
        self._predecessors: dict[str, dict[str, list[str]]] = {}
        self._neighbors: dict[str, dict[str, list[str]]] = {}

    def add_node(self, node):
        self._nodes[node.node_id] = node

    def get_node(self, node_id):
        return self._nodes.get(node_id)

    def add_edge(self, src, tgt, etype, weight=1.0):
        self._neighbors.setdefault(src, {}).setdefault(etype, []).append(tgt)
        self._predecessors.setdefault(tgt, {}).setdefault(etype, []).append(src)

    def neighbors(self, node_id, edge_type=None, direction="out"):
        if edge_type:
            et = edge_type.value if hasattr(edge_type, 'value') else edge_type
            return self._neighbors.get(node_id, {}).get(et, [])
        all_n: list[str] = []
        for lst in self._neighbors.get(node_id, {}).values():
            all_n.extend(lst)
        return all_n

    def predecessors(self, node_id, edge_type=None):
        if edge_type:
            et = edge_type.value if hasattr(edge_type, 'value') else edge_type
            return self._predecessors.get(node_id, {}).get(et, [])
        all_p: list[str] = []
        for lst in self._predecessors.get(node_id, {}).values():
            all_p.extend(lst)
        return all_p

    def successors(self, node_id, edge_type=None):
        return self.neighbors(node_id, edge_type=edge_type, direction="out")

    def nodes(self):
        return list(self._nodes.values())

    def node_count(self):
        return len(self._nodes)


class TestRootCauseDiscoverer:
    @pytest.fixture
    def causal_graph(self):
        g = MockGraph()
        g.add_node(MockNode("root", label="Payment Timeout", confidence=0.9))
        g.add_node(MockNode("mid", label="Checkout Failure", confidence=0.8))
        g.add_node(MockNode("effect", label="Negative Review", confidence=0.7, node_type="problem_signal"))
        g.add_edge("root", "mid", "causes")
        g.add_edge("mid", "effect", "causes")
        return g

    def test_discover_root_causes(self, causal_graph) -> None:
        propagator = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        discoverer = RootCauseDiscoverer(max_depth=5)
        causes = discoverer.discover(causal_graph, propagator=propagator)
        assert len(causes) > 0

    def test_root_cause_has_rank(self, causal_graph) -> None:
        propagator = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        discoverer = RootCauseDiscoverer(max_depth=5)
        causes = discoverer.discover(causal_graph, propagator=propagator)
        assert all(rc.ranking_score >= 0 for rc in causes)

    def test_transitive_impact_counted(self, causal_graph) -> None:
        propagator = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        discoverer = RootCauseDiscoverer(max_depth=5)
        causes = discoverer.discover(causal_graph, propagator=propagator)
        root_causes = [rc for rc in causes if rc.cause_node_id == "root"]
        if root_causes:
            assert root_causes[0].transitive_impact_count >= 1

    def test_confidence_ranking(self, causal_graph) -> None:
        discoverer = RootCauseDiscoverer(ranking=RootCauseRanking.CONFIDENCE, max_depth=5)
        propagator = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        causes = discoverer.discover(causal_graph, propagator=propagator)
        if len(causes) >= 2:
            assert causes[0].ranking_score >= causes[1].ranking_score

    def test_empty_graph(self) -> None:
        g = MockGraph()
        discoverer = RootCauseDiscoverer(max_depth=5)
        propagator = ConfidencePropagator()
        causes = discoverer.discover(g, propagator=propagator)
        assert causes == []

    def test_no_effect_nodes(self) -> None:
        g = MockGraph()
        g.add_node(MockNode("n1", label="Node 1"))
        discoverer = RootCauseDiscoverer(max_depth=5)
        propagator = ConfidencePropagator()
        causes = discoverer.discover(g, propagator=propagator)
        assert causes == []
