"""Tests for ClusterBuilder."""

from __future__ import annotations

from phase2.clustering.builder import ClusterBuilder
from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipEdge, RelationshipGraph
from phase2.clustering.providers.connected_components import ConnectedComponentsProvider


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


class TestClusterBuilder:
    def test_build_cluster(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig()
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)

        cluster = builder.build(["a", "b", "c", "d"], graph)

        assert cluster.member_count == 4
        assert cluster.relationship_count == 6
        assert cluster.provider == "connected_components"
        assert cluster.algorithm == "connected_components"
        assert len(cluster.member_ids) == 4

    def test_deterministic_id(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig()
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)

        c1 = builder.build(["a", "b", "c"], graph)
        c2 = builder.build(["a", "b", "c"], graph)
        assert c1.cluster_id == c2.cluster_id

    def test_different_members_different_ids(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig()
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)

        c1 = builder.build(["a", "b", "c"], graph)
        c2 = builder.build(["a", "b", "d"], graph)
        assert c1.cluster_id != c2.cluster_id

    def test_representative_selection(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig()
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)

        cluster = builder.build(["a", "b", "c", "d"], graph)
        # b has highest weighted_degree: 0.95+0.91+0.85=2.71 vs a: 0.95+0.88+0.82=2.65
        assert cluster.representative_id == "b"

    def test_single_node_cluster(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig()
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)

        cluster = builder.build(["a"], graph)
        assert cluster.member_count == 1
        assert cluster.representative_id == "a"
        assert cluster.relationship_count == 0

    def test_member_ids_sorted(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig()
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)

        cluster = builder.build(["d", "a", "c", "b"], graph)
        assert list(cluster.member_ids) == ["a", "b", "c", "d"]

    def test_quality_score_initially_zero(self) -> None:
        graph = _build_graph()
        config = ClusteringConfig()
        provider = ConnectedComponentsProvider()
        builder = ClusterBuilder(config, provider)

        cluster = builder.build(["a", "b", "c"], graph)
        assert cluster.quality_score == 0.0
