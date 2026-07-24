"""Tests for SimilarityEngine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from phase2.embeddings.schema import SourceType
from phase2.similarity.config import SimilarityEngineConfig
from phase2.similarity.engine import SimilarityEngine


def _create_embedding_parquet(
    path: Path,
    n: int = 10,
    dim: int = 64,
    intra_cluster_similarity: float = 0.95,
) -> None:
    """Create a test embedding parquet file with controllable intra-cluster similarity.

    Uses linear interpolation between cluster bases and random directions.
    intra_cluster_similarity controls how similar cluster-mates are (0.0-1.0).
    """
    rng = np.random.default_rng(42)
    n_clusters = max(2, n // 3)
    bases = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    bases /= np.linalg.norm(bases, axis=1, keepdims=True)

    labels = rng.integers(0, n_clusters, size=n)
    alpha = 1.0 - intra_cluster_similarity  # fraction of random direction

    vecs = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        random_dir = rng.standard_normal(dim).astype(np.float32)
        random_dir /= np.linalg.norm(random_dir)
        vecs[i] = (1 - alpha) * bases[labels[i]] + alpha * random_dir

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    data = {
        "embedding_id": [f"eid_{i}" for i in range(n)],
        "source_id": [f"sid_{i}" for i in range(n)],
        "source_type": ["observation"] * n,
        "provider": ["sentence_transformers"] * n,
        "model": ["all-MiniLM-L6-v2"] * n,
        "model_version": ["fp"] * n,
        "dimension": [dim] * n,
        "embedding": [v.tolist() for v in vecs],
        "text_snippet": [None] * n,
        "created_at": ["2026-01-01"] * n,
    }
    df = pl.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))


class TestSimilarityEngine:
    def test_generate_basic(self, tmp_path: Path) -> None:
        embed_path = tmp_path / "embeddings_observation.parquet"
        _create_embedding_parquet(embed_path, n=20, dim=64, intra_cluster_similarity=0.95)

        cfg = SimilarityEngineConfig(
            similarity_threshold=0.50,
            top_k=5,
            batch_size=10,
            minimum_confidence=0.30,
            output_directory=tmp_path,
            embedding_dimension=64,
            source_paths={SourceType.observation: embed_path},
        )
        engine = SimilarityEngine(cfg)
        result = engine.generate(force=True)

        assert result["total_relationships"] > 0
        assert "elapsed_seconds" in result
        assert "avg_similarity" in result

    def test_generate_skips_existing(self, tmp_path: Path) -> None:
        embed_path = tmp_path / "embeddings_observation.parquet"
        _create_embedding_parquet(embed_path, n=5, dim=64, intra_cluster_similarity=0.95)

        cfg = SimilarityEngineConfig(
            similarity_threshold=0.50,
            top_k=3,
            minimum_confidence=0.30,
            output_directory=tmp_path,
            embedding_dimension=64,
            source_paths={SourceType.observation: embed_path},
        )
        engine = SimilarityEngine(cfg)
        engine.generate(force=True)
        result = engine.generate(force=False)
        assert result.get("status") == "skipped"

    def test_generate_force(self, tmp_path: Path) -> None:
        embed_path = tmp_path / "embeddings_observation.parquet"
        _create_embedding_parquet(embed_path, n=10, dim=64, intra_cluster_similarity=0.95)

        cfg = SimilarityEngineConfig(
            similarity_threshold=0.50,
            top_k=5,
            minimum_confidence=0.30,
            output_directory=tmp_path,
            embedding_dimension=64,
            source_paths={SourceType.observation: embed_path},
        )
        engine = SimilarityEngine(cfg)
        result1 = engine.generate(force=True)
        result2 = engine.generate(force=True)
        assert result2["total_relationships"] == result1["total_relationships"]

    def test_stats(self, tmp_path: Path) -> None:
        embed_path = tmp_path / "embeddings_observation.parquet"
        _create_embedding_parquet(embed_path, n=10, dim=64, intra_cluster_similarity=0.95)

        cfg = SimilarityEngineConfig(
            similarity_threshold=0.50,
            top_k=5,
            minimum_confidence=0.30,
            output_directory=tmp_path,
            embedding_dimension=64,
            source_paths={SourceType.observation: embed_path},
        )
        engine = SimilarityEngine(cfg)
        engine.generate(force=True)
        stats = engine.stats()
        assert stats.total_relationships > 0

    def test_verify(self, tmp_path: Path) -> None:
        embed_path = tmp_path / "embeddings_observation.parquet"
        _create_embedding_parquet(embed_path, n=10, dim=64, intra_cluster_similarity=0.95)

        cfg = SimilarityEngineConfig(
            similarity_threshold=0.50,
            top_k=5,
            minimum_confidence=0.30,
            output_directory=tmp_path,
            embedding_dimension=64,
            source_paths={SourceType.observation: embed_path},
        )
        engine = SimilarityEngine(cfg)
        engine.generate(force=True)
        result = engine.verify()
        assert result["valid"] is True

    def test_search_relationships(self, tmp_path: Path) -> None:
        embed_path = tmp_path / "embeddings_observation.parquet"
        _create_embedding_parquet(embed_path, n=10, dim=64, intra_cluster_similarity=0.95)

        cfg = SimilarityEngineConfig(
            similarity_threshold=0.50,
            top_k=5,
            minimum_confidence=0.30,
            output_directory=tmp_path,
            embedding_dimension=64,
            source_paths={SourceType.observation: embed_path},
        )
        engine = SimilarityEngine(cfg)
        engine.generate(force=True)
        results = engine.search_relationships("sid_0", k=5)
        assert len(results) > 0

    def test_no_embeddings(self, tmp_path: Path) -> None:
        cfg = SimilarityEngineConfig(
            output_directory=tmp_path,
            source_paths={SourceType.observation: tmp_path / "nonexistent.parquet"},
        )
        engine = SimilarityEngine(cfg)
        result = engine.generate(force=True)
        assert result["total_relationships"] == 0
