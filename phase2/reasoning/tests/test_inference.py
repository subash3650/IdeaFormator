"""Tests for InferenceEngine."""

from __future__ import annotations

import pytest

from phase2.reasoning.chains import ChainTracker
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.config import ReasoningConfig
from phase2.reasoning.evidence import EvidenceAggregator
from phase2.reasoning.explanation import ExplanationGenerator
from phase2.reasoning.inference import InferenceEngine
from phase2.reasoning.root_cause import RootCauseDiscoverer
from phase2.reasoning.rule_engine import RuleEngine
from phase2.reasoning.schema import PropagationStrategy
from pathlib import Path


class MockNode:
    def __init__(self, node_id: str, confidence=0.5, label="", node_type=None):
        self.node_id = node_id
        self.confidence = confidence
        self.label = label
        self.node_type = node_type
        self.properties = {}
        self.metadata = {}
        self.attributes = {}


class MockGraph:
    def __init__(self):
        self._nodes: dict[str, MockNode] = {}
        self._edges: list = []
        self._predecessors: dict[str, dict[str, list[str]]] = {}
        self._neighbors: dict[str, dict[str, list[str]]] = {}

    def add_node(self, node):
        self._nodes[node.node_id] = node

    def get_node(self, node_id):
        return self._nodes.get(node_id)

    def add_edge(self, src, tgt, etype, weight=0.5):
        self._edges.append((src, tgt, etype, weight))
        self._neighbors.setdefault(src, {}).setdefault(etype, []).append(tgt)
        self._predecessors.setdefault(tgt, {}).setdefault(etype, []).append(src)

    def neighbors(self, node_id, edge_type=None, direction="out"):
        if edge_type:
            return self._neighbors.get(node_id, {}).get(edge_type.value if hasattr(edge_type, 'value') else edge_type, [])
        all_n: list[str] = []
        for lst in self._neighbors.get(node_id, {}).values():
            all_n.extend(lst)
        return all_n

    def predecessors(self, node_id, edge_type=None):
        if edge_type:
            return self._predecessors.get(node_id, {}).get(edge_type.value if hasattr(edge_type, 'value') else edge_type, [])
        all_p: list[str] = []
        for lst in self._predecessors.get(node_id, {}).values():
            all_p.extend(lst)
        return all_p

    def successors(self, node_id, edge_type=None):
        return self.neighbors(node_id, edge_type=edge_type, direction="out")

    def nodes(self):
        return list(self._nodes.values())

    def edges(self):
        return self._edges

    def node_count(self):
        return len(self._nodes)

    def edge_count(self):
        return len(self._edges)

    def get_edge(self, edge_id):
        return None

    def nodes_by_type(self, ntype):
        return [n for n in self._nodes.values() if n.node_type == ntype]

    def out_degree(self, node_id):
        return len(self._neighbors.get(node_id, {}))

    def in_degree(self, node_id):
        count = 0
        for nid, preds in self._predecessors.items():
            for lst in preds.values():
                if node_id in lst:
                    count += 1
        return count

    def degree(self, node_id):
        return self.out_degree(node_id) + self.in_degree(node_id)


@pytest.fixture
def config():
    return ReasoningConfig(
        output_dir=Path("/tmp/test_reasoning"),
        max_inferences_per_run=500,
        max_chain_length=5,
        max_rule_iterations=3,
        min_confidence=0.1,
        generate_explanations=False,
    )


@pytest.fixture
def graph():
    g = MockGraph()
    # Create a simple causal chain: cause → intermediate → effect
    g.add_node(MockNode("cause", confidence=0.9, label="Root Cause"))
    g.add_node(MockNode("intermediate", confidence=0.8, label="Intermediate"))
    g.add_node(MockNode("effect", confidence=0.7, label="Effect"))
    g.add_edge("cause", "intermediate", "causes")
    g.add_edge("intermediate", "effect", "causes")
    # Create evidence converging on effect
    g.add_node(MockNode("evidence1", confidence=0.85, label="Evidence 1"))
    g.add_node(MockNode("evidence2", confidence=0.75, label="Evidence 2"))
    g.add_edge("evidence1", "effect", "references")
    g.add_edge("evidence2", "effect", "references")
    return g


class TestInferenceEngine:
    def test_infer_returns_output(self, config, graph) -> None:
        rule_engine = RuleEngine(
            enabled_rules=["transitive_closure", "causal_chain", "evidence_convergence"],
            max_iterations=config.max_rule_iterations,
            max_inferences=config.max_inferences_per_run,
        )
        rule_engine.initialize()
        propagator = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        chain_tracker = ChainTracker(run_id="test-run")
        evidence_agg = EvidenceAggregator(min_evidence_count=1)
        root_cause = RootCauseDiscoverer(max_depth=5)
        explanation_gen = ExplanationGenerator()
        engine = InferenceEngine(
            config=config,
            rule_engine=rule_engine,
            propagator=propagator,
            chain_tracker=chain_tracker,
            evidence_aggregator=evidence_agg,
            root_cause_discoverer=root_cause,
            explanation_generator=explanation_gen,
        )
        output = engine.infer(graph, "test-run")
        assert output.inferences is not None
        assert output.chains is not None
        assert output.metadata is not None
        assert output.metadata.run_id == "test-run"

    def test_empty_graph(self, config) -> None:
        empty_graph = MockGraph()
        rule_engine = RuleEngine(
            enabled_rules=["transitive_closure"],
            max_iterations=1,
            max_inferences=100,
        )
        rule_engine.initialize()
        propagator = ConfidencePropagator()
        chain_tracker = ChainTracker()
        evidence_agg = EvidenceAggregator(min_evidence_count=2)
        root_cause = RootCauseDiscoverer()
        explanation_gen = ExplanationGenerator()
        engine = InferenceEngine(
            config=config,
            rule_engine=rule_engine,
            propagator=propagator,
            chain_tracker=chain_tracker,
            evidence_aggregator=evidence_agg,
            root_cause_discoverer=root_cause,
            explanation_generator=explanation_gen,
        )
        output = engine.infer(empty_graph, "empty-run")
        assert isinstance(output.inferences, list)
        assert output.metadata is not None
