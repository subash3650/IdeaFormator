"""Tests for ClusterValidator."""

from __future__ import annotations

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipEdge, RelationshipGraph
from phase2.clustering.schema import ClusterType, SemanticCluster, ValidationIssue
from phase2.clustering.validator import ClusterValidator


def _make_cluster(
    cluster_id: str = "test_cluster",
    representative_id: str = "a",
    members: tuple[str, ...] = ("a", "b", "c"),
) -> SemanticCluster:
    return SemanticCluster(
        cluster_id=cluster_id,
        representative_id=representative_id,
        member_ids=members,
        member_count=len(members),
        relationship_count=len(members) - 1,
        average_similarity=0.9,
        density=0.8,
        quality_score=0.85,
        provider="connected_components",
        provider_version="1.0",
        algorithm="connected_components",
        version="1.0",
    )


def _make_graph(*edges: tuple[str, str]) -> RelationshipGraph:
    g = RelationshipGraph()
    for u, v in edges:
        g.add_edge(RelationshipEdge(source_id=u, target_id=v, similarity=0.9, confidence=0.8))
    return g


class TestClusterValidator:
    def test_valid_cluster_set(self) -> None:
        graph = _make_graph(("a", "b"), ("b", "c"), ("c", "d"))
        config = ClusteringConfig(minimum_cluster_size=2, maximum_cluster_size=10)
        validator = ClusterValidator(config)

        cluster = _make_cluster(members=("a", "b", "c"))
        result = validator.validate([cluster], graph)

        assert result.valid is True
        assert result.clusters_checked == 1

    def test_min_size_violation(self) -> None:
        graph = _make_graph(("a", "b"))
        config = ClusteringConfig(minimum_cluster_size=3)
        validator = ClusterValidator(config)

        cluster = _make_cluster(members=("a", "b"))
        result = validator.validate([cluster], graph)

        assert result.valid is False
        assert any(i.code == "MIN_SIZE_VIOLATION" for i in result.issues)

    def test_max_size_violation(self) -> None:
        graph = RelationshipGraph()
        members = tuple([f"m{i}" for i in range(10)])
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                graph.add_edge(
                    RelationshipEdge(source_id=members[i], target_id=members[j], similarity=0.9, confidence=0.8)
                )

        config = ClusteringConfig(maximum_cluster_size=5)
        validator = ClusterValidator(config)

        cluster = _make_cluster(members=members)
        result = validator.validate([cluster], graph)

        assert result.valid is False
        assert any(i.code == "MAX_SIZE_VIOLATION" for i in result.issues)

    def test_orphan_member(self) -> None:
        graph = _make_graph(("a", "b"))
        config = ClusteringConfig(minimum_cluster_size=2)
        validator = ClusterValidator(config)

        cluster = _make_cluster(members=("a", "b", "orphan"))
        result = validator.validate([cluster], graph)

        assert result.valid is False
        assert any(i.code == "ORPHAN_MEMBER_ID" for i in result.issues)

    def test_unsorted_members(self) -> None:
        graph = _make_graph(("a", "b"), ("b", "c"))
        config = ClusteringConfig(minimum_cluster_size=2)
        validator = ClusterValidator(config)

        # Manually create cluster with unsorted members
        cluster = SemanticCluster(
            cluster_id="test",
            representative_id="b",
            member_ids=("c", "a", "b"),  # Not sorted!
            member_count=3,
            relationship_count=2,
            average_similarity=0.9,
            density=1.0,
            quality_score=0.85,
            provider="cc",
            provider_version="1.0",
            algorithm="cc",
            version="1.0",
        )
        result = validator.validate([cluster], graph)

        assert result.valid is False
        assert any(i.code == "UNSORTED_MEMBERS" for i in result.issues)

    def test_empty_cluster_list(self) -> None:
        graph = _make_graph()
        config = ClusteringConfig()
        validator = ClusterValidator(config)
        result = validator.validate([], graph)
        assert result.valid is True
        assert result.clusters_checked == 0

    def test_overlap_warning(self) -> None:
        graph = _make_graph(("a", "b"), ("b", "c"))
        config = ClusteringConfig(minimum_cluster_size=2)
        validator = ClusterValidator(config)

        c1 = _make_cluster(cluster_id="c1", members=("a", "b"))
        c2 = _make_cluster(cluster_id="c2", members=("b", "c"))
        result = validator.validate([c1, c2], graph)

        # Overlapping members should produce a warning
        assert any(i.code == "OVERLAPPING_MEMBER" for i in result.issues)
