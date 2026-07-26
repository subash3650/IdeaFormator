"""Tests for RuleEngine."""

from __future__ import annotations

from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.rule_engine import RuleEngine
from phase2.reasoning.schema import PropagationStrategy


class MockNode:
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockGraph:
    def __init__(self):
        self._nodes: dict[str, MockNode] = {}
        self._edges: list = []

    def add_node(self, node: MockNode) -> None:
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str):
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
        return self._edges

    def edge_count(self):
        return len(self._edges)

    def get_edge(self, edge_id: str):
        return None

    def nodes_by_type(self, ntype):
        return []

    def out_degree(self, node_id):
        return 0

    def in_degree(self, node_id):
        return 0

    def degree(self, node_id):
        return 0


class TestRuleEngine:
    def test_initialize_with_enabled_rules(self) -> None:
        engine = RuleEngine(
            enabled_rules=["transitive_closure", "causal_chain"],
            max_iterations=3,
            max_inferences=1000,
        )
        engine.initialize()
        names = [r.name for r in engine.rules]
        assert "transitive_closure" in names
        assert "causal_chain" in names
        assert "evidence_convergence" not in names

    def test_initialize_all_rules(self) -> None:
        engine = RuleEngine(max_iterations=3, max_inferences=1000)
        engine.initialize()
        assert len(engine.rules) >= 3

    def test_match_rules_returns_dict(self) -> None:
        engine = RuleEngine(enabled_rules=["transitive_closure"], max_iterations=1, max_inferences=100)
        engine.initialize()
        graph = MockGraph()
        graph.add_node(MockNode("n0"))
        matches = engine.match_rules(graph)
        assert isinstance(matches, dict)
        assert "transitive_closure" in matches

    def test_apply_all_returns_list(self) -> None:
        engine = RuleEngine(enabled_rules=["transitive_closure"], max_iterations=1, max_inferences=100)
        engine.initialize()
        graph = MockGraph()
        graph.add_node(MockNode("n0"))
        propagator = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        results = engine.apply_all(graph, propagator)
        assert isinstance(results, list)

    def test_dependency_resolution(self) -> None:
        engine = RuleEngine(
            enabled_rules=["causal_chain"],
            max_iterations=3,
            max_inferences=1000,
        )
        engine.initialize()
        names = [r.name for r in engine.rules]
        # causal_chain depends on transitive_closure, should be auto-added
        assert "transitive_closure" in names

    def test_respects_max_inferences(self) -> None:
        engine = RuleEngine(
            enabled_rules=["transitive_closure"],
            max_iterations=1,
            max_inferences=1,
        )
        engine.initialize()
        graph = MockGraph()
        for i in range(100):
            graph.add_node(MockNode(f"n{i}"))
        propagator = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        results = engine.apply_all(graph, propagator)
        assert len(results) <= 1
