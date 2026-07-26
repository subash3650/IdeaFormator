"""Tests for ClusteringEngine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.engine import ClusteringEngine


def _create_relationship_parquet(
    path: Path,
    n_clusters: int = 3,
    cluster_size: int = 5,
    similarity: float = 0.95,
) -> None:
    """Create synthetic relationships with clear cluster structure."""
    rows = []
    rng = np.random.default_rng(42)

    for cid in range(n_clusters):
        nodes = [f"cluster{cid}_node{i}" for i in range(cluster_size)]
        for i in range(cluster_size):
            for j in range(i + 1, cluster_size):
                sim = similarity + rng.uniform(-0.05, 0.03)
                sim = max(0.0, min(1.0, sim))
                rows.append({
                    "relationship_id": f"rel_{cid}_{i}_{j}",
                    "source_type": "observation",
                    "source_id": nodes[i],
                    "target_type": "observation",
                    "target_id": nodes[j],
                    "relationship_type": "similar",
                    "similarity_score": sim,
                    "confidence": sim * 0.9,
                    "metric": "cosine",
                    "provider": "cosine",
                    "model_fingerprint": "test",
                    "shared_entities": [],
                    "shared_categories": [],
                    "support_count": 0,
                    "metadata": "{}",
                    "version": "1.0",
                    "created_at": "2026-01-01T00:00:00",
                })

    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))


class TestClusteringEngine:
    def test_generate_basic(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=3, cluster_size=5)

        cfg = ClusteringConfig(
            minimum_cluster_size=3,
            maximum_cluster_size=500,
            relationship_threshold=0.50,
            remove_singletons=True,
            merge_small_clusters=False,
            output_directory=tmp_path,
        )
        engine = ClusteringEngine(cfg)
        result = engine.generate(force=True)

        assert result["total_clusters"] > 0
        assert result["total_members"] > 0
        assert result["elapsed_seconds"] >= 0

    def test_generate_always_overwrites(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=2, cluster_size=4)

        cfg = ClusteringConfig(
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )
        engine = ClusteringEngine(cfg)
        result1 = engine.generate(force=True)
        result2 = engine.generate(force=True)
        # Always overwrites — the second run should produce the same clusters
        assert result2["total_clusters"] == result1["total_clusters"]
        assert result2["total_members"] == result1["total_members"]

    def test_generate_force(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=2, cluster_size=4)

        cfg = ClusteringConfig(
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )
        engine = ClusteringEngine(cfg)
        result1 = engine.generate(force=True)
        result2 = engine.generate(force=True)
        assert result2["total_clusters"] == result1["total_clusters"]

    def test_no_relationships(self, tmp_path: Path) -> None:
        cfg = ClusteringConfig(output_directory=tmp_path)
        engine = ClusteringEngine(cfg)
        result = engine.generate(force=True)
        assert result["total_clusters"] == 0

    def test_stats(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=2, cluster_size=4)

        cfg = ClusteringConfig(
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )
        engine = ClusteringEngine(cfg)
        engine.generate(force=True)
        stats = engine.stats()
        assert stats["total_clusters"] > 0

    def test_verify(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=2, cluster_size=4)

        cfg = ClusteringConfig(
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )
        engine = ClusteringEngine(cfg)
        engine.generate(force=True)
        result = engine.verify()
        assert "valid" in result

    def test_search_clusters(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=2, cluster_size=4)

        cfg = ClusteringConfig(
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )
        engine = ClusteringEngine(cfg)
        engine.generate(force=True)

        # Search by member ID
        results = engine.search_clusters("cluster0_node0")
        assert len(results) >= 1
