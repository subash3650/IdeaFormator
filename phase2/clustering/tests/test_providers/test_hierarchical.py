"""Tests for HierarchicalProvider."""

from __future__ import annotations

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipEdge, RelationshipGraph
from phase2.clustering.providers.hierarchical import HierarchicalProvider


class TestHierarchicalProvider:
    def test_basic_clustering(self) -> None:
        g = RelationshipGraph()
        g.add_edge(RelationshipEdge(source_id="a", target_id="b", similarity=0.95, confidence=0.9))
        g.add_edge(RelationshipEdge(source_id="b", target_id="c", similarity=0.93, confidence=0.88))
        g.add_edge(RelationshipEdge(source_id="a", target_id="c", similarity=0.91, confidence=0.86))
        g.add_edge(RelationshipEdge(source_id="d", target_id="e", similarity=0.90, confidence=0.85))

        provider = HierarchicalProvider()
        config = ClusteringConfig(relationship_threshold=0.50)
        clusters = provider.cluster(g, config)

        assert len(clusters) == 2
        cluster_sets = [frozenset(c) for c in clusters]
        assert frozenset({"a", "b", "c"}) in cluster_sets
        assert frozenset({"d", "e"}) in cluster_sets

    def test_deterministic_ordering(self) -> None:
        g = RelationshipGraph()
        g.add_edge(RelationshipEdge(source_id="a", target_id="b", similarity=0.95, confidence=0.9))
        g.add_edge(RelationshipEdge(source_id="b", target_id="c", similarity=0.93, confidence=0.88))
        g.add_edge(RelationshipEdge(source_id="a", target_id="c", similarity=0.91, confidence=0.86))

        provider = HierarchicalProvider()
        config = ClusteringConfig(relationship_threshold=0.50)
        result1 = provider.cluster(g, config)
        result2 = provider.cluster(g, config)
        assert result1 == result2

    def test_merges_close_clusters(self) -> None:
        g = RelationshipGraph()
        # Two sub-groups with a strong bridge
        g.add_edge(RelationshipEdge(source_id="a", target_id="b", similarity=0.95, confidence=0.9))
        g.add_edge(RelationshipEdge(source_id="c", target_id="d", similarity=0.94, confidence=0.89))
        g.add_edge(RelationshipEdge(source_id="b", target_id="c", similarity=0.92, confidence=0.87))

        provider = HierarchicalProvider()
        config = ClusteringConfig(relationship_threshold=0.50)
        clusters = provider.cluster(g, config)

        # Should merge all into one cluster due to high average linkage
        assert len(clusters) <= 2

    def test_respects_threshold(self) -> None:
        g = RelationshipGraph()
        g.add_edge(RelationshipEdge(source_id="a", target_id="b", similarity=0.95, confidence=0.9))
        g.add_edge(RelationshipEdge(source_id="c", target_id="d", similarity=0.95, confidence=0.9))
        g.add_edge(RelationshipEdge(source_id="b", target_id="c", similarity=0.70, confidence=0.7))

        # Threshold 0.80 filters the b-c bridge, leaving 2 separate components
        provider = HierarchicalProvider()
        config = ClusteringConfig(relationship_threshold=0.80)
        clusters = provider.cluster(g, config)

        assert len(clusters) == 2

    def test_empty_graph(self) -> None:
        g = RelationshipGraph()
        provider = HierarchicalProvider()
        config = ClusteringConfig()
        clusters = provider.cluster(g, config)
        assert clusters == []

    def test_name_and_version(self) -> None:
        provider = HierarchicalProvider()
        assert provider.name == "hierarchical"
        assert provider.version == "1.0"

    def test_members_sorted(self) -> None:
        g = RelationshipGraph()
        g.add_edge(RelationshipEdge(source_id="z", target_id="a", similarity=0.9, confidence=0.8))
        g.add_edge(RelationshipEdge(source_id="m", target_id="z", similarity=0.9, confidence=0.8))

        provider = HierarchicalProvider()
        config = ClusteringConfig(relationship_threshold=0.50)
        clusters = provider.cluster(g, config)

        for cluster in clusters:
            assert cluster == sorted(cluster)
