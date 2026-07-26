"""Tests for SemanticClusterStore."""

from __future__ import annotations

from pathlib import Path

from phase2.clustering.schema import ClusterType, SemanticCluster
from phase2.clustering.store import SemanticClusterStore


def _make_cluster(cluster_id: str = "test_cluster", members: tuple[str, ...] = ("a", "b", "c")) -> SemanticCluster:
    return SemanticCluster(
        cluster_id=cluster_id,
        representative_id=members[0],
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


class TestSemanticClusterStore:
    def test_save_and_load(self, tmp_path: Path) -> None:
        store = SemanticClusterStore(tmp_path)
        clusters = [_make_cluster("c1"), _make_cluster("c2", ("d", "e", "f"))]
        store.save(clusters)
        loaded = store.load()
        assert len(loaded) == 2
        assert loaded[0].cluster_id == "c1"

    def test_exists(self, tmp_path: Path) -> None:
        store = SemanticClusterStore(tmp_path)
        assert not store.exists()
        store.save([_make_cluster()])
        assert store.exists()

    def test_count(self, tmp_path: Path) -> None:
        store = SemanticClusterStore(tmp_path)
        store.save([_make_cluster("c1"), _make_cluster("c2")])
        assert store.count() == 2

    def test_overwrite(self, tmp_path: Path) -> None:
        store = SemanticClusterStore(tmp_path)
        store.save([_make_cluster("old")])
        store.save([_make_cluster("new")])
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].cluster_id == "new"

    def test_load_df(self, tmp_path: Path) -> None:
        store = SemanticClusterStore(tmp_path)
        store.save([_make_cluster()])
        df = store.load_df()
        assert df.height == 1

    def test_load_empty(self, tmp_path: Path) -> None:
        store = SemanticClusterStore(tmp_path)
        loaded = store.load()
        assert loaded == []

    def test_roundtrip(self, tmp_path: Path) -> None:
        store = SemanticClusterStore(tmp_path)
        original = _make_cluster(
            "roundtrip",
            members=("x", "y", "z"),
        )
        store.save([original])
        loaded = store.load()[0]
        assert loaded.cluster_id == original.cluster_id
        assert loaded.member_ids == original.member_ids
        assert loaded.representative_id == original.representative_id
        assert loaded.quality_score == original.quality_score
