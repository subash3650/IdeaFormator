"""Tests for deterministic outputs — running clustering twice must produce identical results."""

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


class TestDeterminism:
    def test_identical_cluster_ids(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=3, cluster_size=5)

        cfg = ClusteringConfig(
            minimum_cluster_size=3,
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )

        # Run 1
        engine1 = ClusteringEngine(cfg)
        result1 = engine1.generate(force=True)
        clusters1 = sorted(engine1.search_clusters("cluster0_node0"), key=lambda c: c.cluster_id)

        # Run 2
        engine2 = ClusteringEngine(cfg)
        result2 = engine2.generate(force=True)
        clusters2 = sorted(engine2.search_clusters("cluster0_node0"), key=lambda c: c.cluster_id)

        # IDs must be identical
        ids1 = [c.cluster_id for c in clusters1]
        ids2 = [c.cluster_id for c in clusters2]
        assert ids1 == ids2, f"Cluster IDs differ: {ids1} vs {ids2}"

    def test_identical_representatives(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=3, cluster_size=5)

        cfg = ClusteringConfig(
            minimum_cluster_size=3,
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )

        engine1 = ClusteringEngine(cfg)
        engine1.generate(force=True)
        reprs1 = sorted(
            [c.representative_id for c in engine1.search_clusters("cluster0_node0")]
        )

        engine2 = ClusteringEngine(cfg)
        engine2.generate(force=True)
        reprs2 = sorted(
            [c.representative_id for c in engine2.search_clusters("cluster0_node0")]
        )

        assert reprs1 == reprs2, f"Representatives differ: {reprs1} vs {reprs2}"

    def test_identical_cluster_count(self, tmp_path: Path) -> None:
        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, n_clusters=2, cluster_size=4)

        cfg = ClusteringConfig(
            minimum_cluster_size=3,
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )

        engine1 = ClusteringEngine(cfg)
        result1 = engine1.generate(force=True)

        engine2 = ClusteringEngine(cfg)
        result2 = engine2.generate(force=True)

        assert result1["total_clusters"] == result2["total_clusters"]
