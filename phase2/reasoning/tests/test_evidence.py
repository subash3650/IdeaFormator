"""Tests for EvidenceAggregator."""

from __future__ import annotations

from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.evidence import EvidenceAggregator
from phase2.reasoning.schema import PropagationStrategy


class MockNode:
    def __init__(self, node_id: str, confidence: float = 0.5, label: str = ""):
        self.node_id = node_id
        self.confidence = confidence
        self.label = label


class MockGraph:
    def __init__(self):
        self._nodes: dict[str, MockNode] = {}
        self._predecessors: dict[str, dict[str, list[str]]] = {}
        self._successors: dict[str, list[str]] = {}

    def add_node(self, node: MockNode) -> None:
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str):
        return self._nodes.get(node_id)

    def add_edge(self, src: str, tgt: str, etype: str) -> None:
        if etype not in self._predecessors:
            self._predecessors.setdefault(tgt, {}).setdefault(etype, []).append(src)
        else:
            self._predecessors.setdefault(tgt, {}).setdefault(etype, []).append(src)
        self._successors.setdefault(src, []).append(tgt)

    def predecessors(self, node_id: str, edge_type=None):
        if edge_type is None:
            result: list[str] = []
            for lst in self._predecessors.get(node_id, {}).values():
                result.extend(lst)
            return result
        et = edge_type.value if hasattr(edge_type, 'value') else edge_type
        return self._predecessors.get(node_id, {}).get(et, [])

    def successors(self, node_id: str, edge_type=None):
        return self._successors.get(node_id, [])

    def neighbors(self, node_id: str, edge_type=None, direction="out"):
        return self._successors.get(node_id, [])

    def nodes(self):
        return list(self._nodes.values())

    def node_count(self):
        return len(self._nodes)


class TestEvidenceAggregator:
    def test_aggregate_single_path(self) -> None:
        graph = MockGraph()
        graph.add_node(MockNode("target", label="Target"))
        graph.add_node(MockNode("src1", confidence=0.8))
        graph.add_edge("src1", "target", "references")
        agg = EvidenceAggregator(min_evidence_count=1)
        propagator = ConfidencePropagator()
        paths = [["src1", "target"]]
        result = agg.aggregate(graph, "target", paths, propagator)
        assert result.conclusion_node_id == "target"
        assert result.evidence_count == 1
        assert result.aggregated_confidence > 0

    def test_aggregate_multiple_paths(self) -> None:
        graph = MockGraph()
        graph.add_node(MockNode("target", label="Target"))
        graph.add_node(MockNode("src1", confidence=0.9))
        graph.add_node(MockNode("src2", confidence=0.7))
        graph.add_edge("src1", "target", "references")
        graph.add_edge("src2", "target", "supported_by")
        agg = EvidenceAggregator(min_evidence_count=2)
        propagator = ConfidencePropagator()
        paths = [["src1", "target"], ["src2", "target"]]
        result = agg.aggregate(graph, "target", paths, propagator)
        assert result.evidence_count == 2
        assert result.aggregated_confidence > 0

    def test_aggregate_all_empty(self) -> None:
        graph = MockGraph()
        graph.add_node(MockNode("n1"))
        agg = EvidenceAggregator(min_evidence_count=2)
        propagator = ConfidencePropagator()
        results = agg.aggregate_all(graph, propagator)
        assert results == []

    def test_find_conflicting(self) -> None:
        graph = MockGraph()
        graph.add_node(MockNode("target", confidence=0.8))
        graph.add_node(MockNode("src1", confidence=0.2))
        graph.add_edge("src1", "target", "references")
        agg = EvidenceAggregator(conflicting_threshold=0.3)
        conflicting = agg.find_conflicting(graph, "target")
        assert len(conflicting) > 0
