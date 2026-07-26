"""Tests for ClusterEvaluator."""

from __future__ import annotations

from phase2.clustering.builder import ClusterBuilder
from phase2.clustering.config import ClusteringConfig
from phase2.clustering.engine import _load_relationship_graph
from phase2.clustering.evaluator import ClusterEvaluator
from phase2.clustering.graph import RelationshipEdge, RelationshipGraph
from phase2.clustering.providers.connected_components import ConnectedComponentsProvider
from phase2.clustering.schema import ClusterType


def _build_graph() -> RelationshipGraph:
    """Build a fully connected 4-node graph."""
    edges = [
        RelationshipEdge(source_id="a", target_id="b", similarity=0.95, confidence=0.9),
        RelationshipEdge(source_id="a", target_id="c", similarity=0.88, confidence=0.8),
        RelationshipEdge(source_id="a", target_id="d", similarity=0.82, confidence=0.75),
        RelationshipEdge(source_id="b", target_id="c", similarity=0.91, confidence=0.85),
        RelationshipEdge(source_id="b", target_id="d", similarity=0.85, confidence=0.8),
        RelationshipEdge(source_id="c", target_id="d", similarity=0.87, confidence=0.82),
    ]
    return RelationshipGraph.from_edges(edges)


class TestClusterEvaluator:
    def test_evaluate_dense_cluster(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig(quality_threshold=0.50)
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)
        evaluator = ClusterEvaluator(config)

        cluster = builder.build(["a", "b", "c", "d"], graph)
        metrics = evaluator.evaluate(cluster, graph)

        assert metrics.member_count == 4
        assert metrics.internal_edge_count == 6
        assert metrics.external_edge_count == 0
        assert 0.0 <= metrics.quality_score <= 1.0
        assert 0.0 <= metrics.internal_cohesion <= 1.0
        assert 0.0 <= metrics.density <= 1.0

    def test_evaluate_and_update(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig(quality_threshold=0.50)
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)
        evaluator = ClusterEvaluator(config)

        cluster = builder.build(["a", "b", "c", "d"], graph)
        updated = evaluator.evaluate_and_update(cluster, graph)

        assert updated.quality_score > 0.0
        assert updated.cluster_type == ClusterType.NORMAL

    def test_low_quality_classification(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig(quality_threshold=0.99)
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)
        evaluator = ClusterEvaluator(config)

        cluster = builder.build(["a", "b", "c", "d"], graph)
        updated = evaluator.evaluate_and_update(cluster, graph)

        assert updated.cluster_type == ClusterType.LOW_QUALITY

    def test_external_separation(self) -> None:
        g = RelationshipGraph()
        g.add_edge(RelationshipEdge(source_id="a", target_id="b", similarity=0.95, confidence=0.9))
        g.add_edge(RelationshipEdge(source_id="a", target_id="c", similarity=0.50, confidence=0.5))
        g.add_edge(RelationshipEdge(source_id="b", target_id="c", similarity=0.50, confidence=0.5))

        config = ClusteringConfig(quality_threshold=0.30)
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)
        evaluator = ClusterEvaluator(config)

        # Only a-b are in cluster, c is external
        cluster = builder.build(["a", "b"], g)
        metrics = evaluator.evaluate(cluster, g)

        assert metrics.external_edge_count == 2  # a-c and b-c
        assert metrics.external_separation <= 1.0

    def test_single_node_cluster(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig()
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)
        evaluator = ClusterEvaluator(config)

        cluster = builder.build(["a"], graph)
        metrics = evaluator.evaluate(cluster, graph)

        assert metrics.member_count == 1
        assert metrics.internal_edge_count == 0
        assert metrics.external_edge_count == 3
        assert metrics.quality_score >= 0.0
