"""Tests for ConnectedComponentsProvider."""

from __future__ import annotations

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipEdge, RelationshipGraph
from phase2.clustering.providers.connected_components import ConnectedComponentsProvider


class TestConnectedComponentsProvider:
    def test_basic_clustering(self) -> None:
        g = RelationshipGraph()
        g.add_edge(RelationshipEdge(source_id="a", target_id="b", similarity=0.9, confidence=0.8))
        g.add_edge(RelationshipEdge(source_id="b", target_id="c", similarity=0.9, confidence=0.8))
        g.add_edge(RelationshipEdge(source_id="d", target_id="e", similarity=0.9, confidence=0.8))

        provider = ConnectedComponentsProvider()
        config = ClusteringConfig()
        clusters = provider.cluster(g, config)

        assert len(clusters) == 2
        cluster_sets = [frozenset(c) for c in clusters]
        assert frozenset({"a", "b", "c"}) in cluster_sets
        assert frozenset({"d", "e"}) in cluster_sets

    def test_deterministic_ordering(self) -> None:
        g = RelationshipGraph()
        for i in range(10):
            for j in range(i + 1, 10):
                g.add_edge(RelationshipEdge(source_id=f"n{i}", target_id=f"n{j}", similarity=0.9, confidence=0.8))

        provider = ConnectedComponentsProvider()
        config = ClusteringConfig()
        result1 = provider.cluster(g, config)
        result2 = provider.cluster(g, config)

        assert result1 == result2

    def test_empty_graph(self) -> None:
        g = RelationshipGraph()
        provider = ConnectedComponentsProvider()
        config = ClusteringConfig()
        clusters = provider.cluster(g, config)
        assert clusters == []

    def test_single_component(self) -> None:
        g = RelationshipGraph()
        g.add_edge(RelationshipEdge(source_id="a", target_id="b", similarity=0.9, confidence=0.8))
        g.add_edge(RelationshipEdge(source_id="b", target_id="c", similarity=0.9, confidence=0.8))
        g.add_edge(RelationshipEdge(source_id="c", target_id="d", similarity=0.9, confidence=0.8))

        provider = ConnectedComponentsProvider()
        config = ClusteringConfig()
        clusters = provider.cluster(g, config)
        assert len(clusters) == 1
        assert len(clusters[0]) == 4

    def test_many_components(self) -> None:
        g = RelationshipGraph()
        for i in range(5):
            g.add_edge(RelationshipEdge(source_id=f"s{i}_0", target_id=f"s{i}_1", similarity=0.9, confidence=0.8))

        provider = ConnectedComponentsProvider()
        config = ClusteringConfig()
        clusters = provider.cluster(g, config)
        assert len(clusters) == 5

    def test_name_and_version(self) -> None:
        provider = ConnectedComponentsProvider()
        assert provider.name == "connected_components"
        assert provider.version == "1.0"

    def test_members_sorted(self) -> None:
        g = RelationshipGraph()
        g.add_edge(RelationshipEdge(source_id="z", target_id="a", similarity=0.9, confidence=0.8))
        g.add_edge(RelationshipEdge(source_id="m", target_id="z", similarity=0.9, confidence=0.8))

        provider = ConnectedComponentsProvider()
        config = ClusteringConfig()
        clusters = provider.cluster(g, config)

        for cluster in clusters:
            assert cluster == sorted(cluster)
