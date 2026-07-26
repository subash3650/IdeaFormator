"""Tests for ExplanationGenerator."""

from __future__ import annotations

from phase2.reasoning.explanation import ExplanationGenerator
from phase2.reasoning.schema import (
    ExplanationFormat,
    InferenceResult,
    InferenceType,
    ProvenanceVersion,
    ReasoningChain,
    ReasoningStep,
)


class MockNode:
    def __init__(self, node_id: str, label: str = "", confidence: float = 0.5):
        self.node_id = node_id
        self.label = label
        self.confidence = confidence


class MockGraph:
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


class TestExplanationGenerator:
    def test_explain_transitive_inference(self) -> None:
        graph = MockGraph()
        graph.add_node(MockNode("n1", label="Root", confidence=0.9))
        graph.add_node(MockNode("n2", label="Middle", confidence=0.8))
        graph.add_node(MockNode("n3", label="Target", confidence=0.7))

        inf = InferenceResult(
            inference_id="inf1",
            inference_type=InferenceType.TRANSITIVE,
            confidence=0.72,
            chain_id="chain1",
            provenance=["n1", "n2", "n3"],
        )
        step = ReasoningStep(
            step_id=0,
            rule_name="transitive_closure",
            input_node_ids=["n1", "n2", "n3"],
            output_edge_id="causes:n1:n3",
            confidence_delta=0.72,
        )
        chain = ReasoningChain(
            chain_id="chain1",
            inference_id="inf1",
            steps=[step],
            input_node_ids=["n1", "n2", "n3"],
            total_confidence=0.72,
            provenance_version=ProvenanceVersion(),
        )

        gen = ExplanationGenerator()
        exp = gen.explain_inference(inf, chain, graph, format=ExplanationFormat.TEMPLATE)
        assert "Transitive" in exp.title
        assert exp.raw_text != ""
        assert exp.explainability_score is not None
        assert exp.explainability_score.confidence == 0.72

    def test_explain_evidence_aggregation(self) -> None:
        graph = MockGraph()
        graph.add_node(MockNode("target", label="Conclusion"))
        graph.add_node(MockNode("e1", label="Evidence 1"))

        inf = InferenceResult(
            inference_id="inf2",
            inference_type=InferenceType.EVIDENCE_AGGREGATION,
            derived_node_id="target",
            confidence=0.85,
            chain_id="chain2",
            provenance=["e1", "target"],
        )
        chain = ReasoningChain(
            chain_id="chain2",
            inference_id="inf2",
            input_node_ids=["e1", "target"],
            total_confidence=0.85,
        )
        gen = ExplanationGenerator()
        exp = gen.explain_inference(inf, chain, graph)
        assert "Evidence" in exp.title
        assert exp.summary != ""

    def test_collapse_long_chain(self) -> None:
        graph = MockGraph()
        for i in range(10):
            graph.add_node(MockNode(f"n{i}", label=f"Node {i}"))

        steps = [
            ReasoningStep(step_id=i, rule_name="rule",
                         input_node_ids=[f"n{i}"], output_node_id=f"n{i+1}", confidence_delta=0.9)
            for i in range(8)
        ]
        inf = InferenceResult(
            inference_id="inf3",
            inference_type=InferenceType.TRANSITIVE,
            confidence=0.5,
            chain_id="chain3",
            provenance=[f"n{i}" for i in range(9)],
        )
        chain = ReasoningChain(
            chain_id="chain3",
            inference_id="inf3",
            steps=steps,
            input_node_ids=[f"n{i}" for i in range(9)],
            total_confidence=0.5,
        )
        gen = ExplanationGenerator()
        exp = gen.explain_inference(inf, chain, graph, collapse_threshold=3)
        assert exp.collapsed_step_count > 0
        assert any("intermediate" in s for s in exp.steps)

    def test_explain_root_cause(self) -> None:
        graph = MockGraph()
        graph.add_node(MockNode("cause", label="Root Cause"))
        graph.add_node(MockNode("effect", label="Problem Signal"))

        from phase2.reasoning.schema import RootCause, RootCauseRanking
        rc = RootCause(
            cause_node_id="cause",
            cause_label="Root Cause",
            effect_node_id="effect",
            effect_label="Problem Signal",
            path=["cause", "mid", "effect"],
            path_length=2,
            propagated_confidence=0.75,
            transitive_impact_count=3,
            evidence_count=2,
            ranking_score=0.9,
        )
        gen = ExplanationGenerator()
        exp = gen.explain_root_cause(rc, graph)
        assert "Root Cause" in exp.title
        assert exp.raw_text != ""
