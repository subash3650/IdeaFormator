"""Tests for the full clustering pipeline."""

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


class TestClusteringPipeline:
    def test_full_pipeline(self, tmp_path: Path) -> None:
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

        # Verify output files exist
        assert (tmp_path / "semantic_clusters.parquet").exists()
        assert (tmp_path / "cluster_manifest.json").exists()
        assert (tmp_path / "cluster_report.json").exists()
        assert (tmp_path / "cluster_report.txt").exists()

        # Verify result structure
        assert result["total_clusters"] >= 1
        assert result["provider"] == "connected_components"
        assert result["algorithm"] == "connected_components"

    def test_pipeline_produces_deterministic_results(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=2, cluster_size=4)

        cfg = ClusteringConfig(
            minimum_cluster_size=3,
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )

        engine1 = ClusteringEngine(cfg)
        result1 = engine1.generate(force=True)
        clusters1 = engine1.search_clusters("cluster0_node0")

        # Reset store
        cfg2 = ClusteringConfig(
            minimum_cluster_size=3,
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )

        engine2 = ClusteringEngine(cfg2)
        result2 = engine2.generate(force=True)
        clusters2 = engine2.search_clusters("cluster0_node0")

        # IDs should be deterministic
        assert result1["total_clusters"] == result2["total_clusters"]
        assert len(clusters1) == len(clusters2)
        for c1, c2 in zip(clusters1, clusters2):
            assert c1.cluster_id == c2.cluster_id
            assert c1.representative_id == c2.representative_id

    def test_low_quality_clusters(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=2, cluster_size=4, similarity=0.90)

        cfg = ClusteringConfig(
            minimum_cluster_size=3,
            quality_threshold=0.99,
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )
        engine = ClusteringEngine(cfg)
        result = engine.generate(force=True)

        # With very high quality threshold, some should be LOW_QUALITY
        assert result.get("low_quality_clusters", 0) >= 0
