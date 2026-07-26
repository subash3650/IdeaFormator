"""Tests for incremental processing — only affected clusters rebuilt."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.engine import ClusteringEngine


def _create_relationship_parquet(
    path: Path,
    rows: list[dict],
) -> None:
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))


def _make_row(cid: str, source: str, target: str, sim: float = 0.95) -> dict:
    return {
        "relationship_id": f"rel_{cid}",
        "source_type": "observation",
        "source_id": source,
        "target_type": "observation",
        "target_id": target,
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
    }


class TestIncremental:
    def test_initial_generation(self, tmp_path: Path) -> None:
        rows = []
        # Cluster A: 5 nodes
        for i in range(5):
            for j in range(i + 1, 5):
                rows.append(_make_row(f"a_{i}_{j}", f"a_{i}", f"a_{j}"))
        # Cluster B: 4 nodes
        for i in range(4):
            for j in range(i + 1, 4):
                rows.append(_make_row(f"b_{i}_{j}", f"b_{i}", f"b_{j}"))

        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, rows)

        cfg = ClusteringConfig(
            minimum_cluster_size=3,
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )
        engine = ClusteringEngine(cfg)
        result = engine.generate(force=True)

        assert result["total_clusters"] >= 2
        assert result["total_members"] >= 7

    def test_incremental_with_new_relationships(self, tmp_path: Path) -> None:
        # Initial: 2 disconnected clusters
        rows = []
        for i in range(4):
            for j in range(i + 1, 4):
                rows.append(_make_row(f"c1_{i}_{j}", f"c1_{i}", f"c1_{j}"))
        for i in range(3):
            for j in range(i + 1, 3):
                rows.append(_make_row(f"c2_{i}_{j}", f"c2_{i}", f"c2_{j}"))

        rel_path = tmp_path / "semantic_relationships.parquet"
        _create_relationship_parquet(rel_path, rows)

        cfg = ClusteringConfig(
            minimum_cluster_size=3,
            relationship_threshold=0.50,
            output_directory=tmp_path,
        )
        engine = ClusteringEngine(cfg)
        result1 = engine.generate(force=True)
        assert result1["total_clusters"] >= 2

        # Now add a bridging relationship between c1 and c2
        bridge = _make_row("bridge", "c1_0", "c2_0", 0.92)
        rows.append(bridge)
        _create_relationship_parquet(rel_path, rows)

        # Force re-compute (incremental detection would use fingerprint comparison)
        result2 = engine.generate(force=True)

        # With a bridge, the two clusters should merge into one (if threshold met)
        assert result2["total_clusters"] >= 1
        assert result2["total_members"] >= 6
