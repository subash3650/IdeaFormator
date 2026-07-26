"""Tests for ConfidencePropagator."""

from __future__ import annotations

import pytest

from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.schema import PropagationStrategy


class MockGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}

    def add_node(self, node_id: str, confidence: float) -> None:
        self._nodes[node_id] = {"confidence": confidence}

    def get_node(self, node_id: str):
        data = self._nodes.get(node_id)
        if data:
            class MockNode:
                def __init__(self, conf):
                    self.confidence = conf
            return MockNode(data["confidence"])
        return None


@pytest.fixture
def graph():
    g = MockGraph()
    for i in range(5):
        g.add_node(f"n{i}", confidence=0.5 + i * 0.1)
    return g


class TestConfidencePropagator:
    def test_multiplicative(self, graph) -> None:
        p = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        path = ["n0", "n1", "n2"]
        conf = p.propagate(path, graph)
        expected = 0.5 * 0.6 * 0.6 * 0.7
        assert conf == pytest.approx(expected, rel=1e-4)

    def test_multiplicative_short_path(self, graph) -> None:
        p = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        conf = p.propagate(["n0"], graph)
        assert conf == 0.0

    def test_minimum(self, graph) -> None:
        p = ConfidencePropagator(strategy=PropagationStrategy.MINIMUM)
        path = ["n0", "n1", "n2"]
        conf = p.propagate(path, graph)
        assert conf == min(0.5, 0.6, 0.6, 0.7)

    def test_decay(self, graph) -> None:
        p = ConfidencePropagator(strategy=PropagationStrategy.DECAY, decay_rate=0.9)
        path = ["n0", "n1", "n2"]
        conf = p.propagate(path, graph)
        expected = 0.5 * (0.9 ** 2)
        assert conf == pytest.approx(expected, rel=1e-4)

    def test_weighted_average(self, graph) -> None:
        p = ConfidencePropagator(strategy=PropagationStrategy.WEIGHTED_AVERAGE)
        path = ["n0", "n1", "n2"]
        conf = p.propagate(path, graph)
        assert 0.5 <= conf <= 0.7

    def test_aggregate_basic(self) -> None:
        p = ConfidencePropagator()
        result = p.aggregate([0.5, 0.7, 0.9])
        assert result == pytest.approx(0.7, rel=1e-4)

    def test_aggregate_with_weights(self) -> None:
        p = ConfidencePropagator()
        result = p.aggregate([0.5, 0.9], weights=[1.0, 3.0])
        assert result == pytest.approx((0.5 + 2.7) / 4.0, rel=1e-4)

    def test_aggregate_empty(self) -> None:
        p = ConfidencePropagator()
        assert p.aggregate([]) == 0.0

    def test_aggregate_zero_weight(self) -> None:
        p = ConfidencePropagator()
        assert p.aggregate([0.5, 0.9], weights=[0.0, 0.0]) == 0.0

    def test_compute_path_confidence(self) -> None:
        p = ConfidencePropagator(strategy=PropagationStrategy.MULTIPLICATIVE)
        result = p.compute_path_confidence([0.8, 0.7, 0.6], [0.9, 0.8])
        expected = 0.8 * 0.7 * 0.6 * 0.9 * 0.8
        assert result == pytest.approx(expected, rel=1e-4)

    def test_above_threshold(self) -> None:
        p = ConfidencePropagator(min_confidence=0.15)
        assert p.above_threshold(0.2) is True
        assert p.above_threshold(0.1) is False

    def test_missing_node(self) -> None:
        p = ConfidencePropagator()
        empty_graph = MockGraph()
        conf = p.propagate(["n0", "n1"], empty_graph)
        assert conf > 0

    def test_strategy_property(self) -> None:
        p = ConfidencePropagator(strategy=PropagationStrategy.DECAY)
        assert p.strategy == PropagationStrategy.DECAY
