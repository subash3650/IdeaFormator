"""Tests that reasoning does not modify the knowledge graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase2.reasoning.chains import ChainTracker
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.evidence import EvidenceAggregator
from phase2.reasoning.explanation import ExplanationGenerator
from phase2.reasoning.root_cause import RootCauseDiscoverer
from phase2.reasoning.rule_engine import RuleEngine
from phase2.reasoning.schema import PropagationStrategy


class MockNode:
    def __init__(self, node_id: str, confidence: float = 0.5, node_type=None):
        self.node_id = node_id
        self.confidence = confidence
        self.node_type = node_type
        self.label = node_id


class SimpleMockGraph:
    def __init__(self):
        self._nodes: dict[str, MockNode] = {}

    def add_node(self, node):
        self._nodes[node.node_id] = node

    def get_node(self, node_id):
        return self._nodes.get(node_id)

    def nodes(self):
        return list(self._nodes.values())

    def node_count(self):
        return len(self._nodes)

    def neighbors(self, node_id, edge_type=None, direction="out"):
        return []

    def predecessors(self, node_id, edge_type=None):
        return []

    def successors(self, node_id, edge_type=None):
        return []

    def edges(self):
        return []

    def edge_count(self):
        return 0

    def get_edge(self, edge_id):
        return None

    def nodes_by_type(self, ntype):
        return []

    def out_degree(self, node_id):
        return 0

    def in_degree(self, node_id):
        return 0

    def degree(self, node_id):
        return 0


class TestNoModification:
    def test_rule_applied_does_not_mutate_graph(self) -> None:
        graph = SimpleMockGraph()
        graph.add_node(MockNode("n1", confidence=0.8))
        graph.add_node(MockNode("n2", confidence=0.7))
        initial_count = graph.node_count()

        engine = RuleEngine(enabled_rules=["transitive_closure"], max_iterations=1, max_inferences=100)
        engine.initialize()
        propagator = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        engine.apply_all(graph, propagator)

        assert graph.node_count() == initial_count

    def test_root_cause_discovery_does_not_mutate(self) -> None:
        graph = SimpleMockGraph()
        graph.add_node(MockNode("n1"))
        graph.add_node(MockNode("n2"))
        initial_count = graph.node_count()

        discoverer = RootCauseDiscoverer(max_depth=5)
        propagator = ConfidencePropagator()
        discoverer.discover(graph, propagator=propagator)

        assert graph.node_count() == initial_count

    def test_evidence_aggregation_does_not_mutate(self) -> None:
        graph = SimpleMockGraph()
        graph.add_node(MockNode("n1"))
        initial_count = graph.node_count()

        agg = EvidenceAggregator(min_evidence_count=2)
        propagator = ConfidencePropagator()
        agg.aggregate_all(graph, propagator)

        assert graph.node_count() == initial_count
