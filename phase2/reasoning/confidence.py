"""Confidence propagation strategies for the Reasoning Engine."""

from __future__ import annotations

import math

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.reasoning.schema import PropagationStrategy


class ConfidencePropagator:
    def __init__(
        self,
        strategy: PropagationStrategy = PropagationStrategy.MULTIPLICATIVE,
        decay_rate: float = 0.9,
        min_confidence: float = 0.15,
    ) -> None:
        self._strategy = strategy
        self._decay_rate = max(0.0, min(1.0, decay_rate))
        self._min_confidence = max(0.0, min(1.0, min_confidence))

    @property
    def strategy(self) -> PropagationStrategy:
        return self._strategy

    def propagate(self, path: list[str], graph: GraphInterface) -> float:
        if len(path) < 2:
            return 0.0

        if self._strategy == PropagationStrategy.MULTIPLICATIVE:
            result = 1.0
            for i in range(len(path) - 1):
                node = graph.get_node(path[i])
                neighbor = graph.get_node(path[i + 1])
                node_conf = node.confidence if node else 0.5
                neighbor_conf = neighbor.confidence if neighbor else 0.5
                result *= node_conf * neighbor_conf
            return max(0.0, min(1.0, result))

        if self._strategy == PropagationStrategy.MINIMUM:
            values: list[float] = []
            for i in range(len(path) - 1):
                node = graph.get_node(path[i])
                neighbor = graph.get_node(path[i + 1])
                node_conf = node.confidence if node else 0.5
                neighbor_conf = neighbor.confidence if neighbor else 0.5
                values.append(node_conf)
                values.append(neighbor_conf)
            return max(0.0, min(1.0, min(values) if values else 0.0))

        if self._strategy == PropagationStrategy.DECAY:
            start_node = graph.get_node(path[0])
            start_conf = start_node.confidence if start_node else 0.5
            depth = len(path) - 1
            decayed = start_conf * (self._decay_rate ** depth)
            return max(0.0, min(1.0, decayed))

        if self._strategy == PropagationStrategy.WEIGHTED_AVERAGE:
            total_weight = 0.0
            weighted_sum = 0.0
            for i in range(len(path) - 1):
                node = graph.get_node(path[i])
                neighbor = graph.get_node(path[i + 1])
                node_conf = node.confidence if node else 0.5
                neighbor_conf = neighbor.confidence if neighbor else 0.5
                weight = 1.0 / (i + 1.0)
                weighted_sum += node_conf * weight * 0.5
                weighted_sum += neighbor_conf * weight * 0.5
                total_weight += weight
            if total_weight == 0:
                return 0.0
            return max(0.0, min(1.0, weighted_sum / total_weight))

        return 0.0

    def aggregate(self, confidences: list[float], weights: list[float] | None = None) -> float:
        if not confidences:
            return 0.0
        if weights is None:
            weights = [1.0] * len(confidences)
        if len(weights) != len(confidences):
            weights = [1.0] * len(confidences)
        total_weight = sum(abs(w) for w in weights)
        if total_weight == 0:
            return 0.0
        weighted = sum(c * w for c, w in zip(confidences, weights))
        return max(0.0, min(1.0, weighted / total_weight))

    def compute_path_confidence(
        self,
        node_confs: list[float],
        edge_weights: list[float] | None = None,
    ) -> float:
        if not node_confs or len(node_confs) < 2:
            return 0.0
        if edge_weights is None:
            edge_weights = [1.0] * (len(node_confs) - 1)
        if len(edge_weights) != len(node_confs) - 1:
            edge_weights = [1.0] * (len(node_confs) - 1)

        if self._strategy == PropagationStrategy.MULTIPLICATIVE:
            result = 1.0
            for i in range(len(node_confs)):
                result *= node_confs[i]
                if i < len(edge_weights):
                    result *= edge_weights[i]
            return max(0.0, min(1.0, result))

        if self._strategy == PropagationStrategy.MINIMUM:
            all_vals = node_confs + edge_weights
            return max(0.0, min(1.0, min(all_vals)))

        if self._strategy == PropagationStrategy.DECAY:
            depth = len(node_confs) - 1
            decayed = node_confs[0] * (self._decay_rate ** depth)
            return max(0.0, min(1.0, decayed))

        if self._strategy == PropagationStrategy.WEIGHTED_AVERAGE:
            total = 0.0
            count = 0
            for i, nc in enumerate(node_confs):
                total += nc
                count += 1
                if i < len(edge_weights):
                    total += edge_weights[i]
                    count += 1
            return max(0.0, min(1.0, total / count if count > 0 else 0.0))

        return 0.0

    def above_threshold(self, confidence: float) -> bool:
        return confidence >= self._min_confidence
