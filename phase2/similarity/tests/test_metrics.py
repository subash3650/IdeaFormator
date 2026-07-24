"""Tests for metrics computation."""

from __future__ import annotations

import pytest
import polars as pl

from phase2.similarity.metrics import RelationshipStats, compute_stats


class TestComputeStats:
    def _make_df(self, rows: list[dict]) -> pl.DataFrame:
        schema = {
            "relationship_id": str,
            "source_type": str,
            "source_id": str,
            "target_type": str,
            "target_id": str,
            "relationship_type": str,
            "similarity_score": float,
            "confidence": float,
            "metric": str,
            "provider": str,
            "model_fingerprint": str,
            "shared_entities": list,
            "shared_categories": list,
            "support_count": int,
            "metadata": str,
            "version": str,
            "created_at": str,
        }
        return pl.DataFrame(rows, schema=schema)

    def test_empty_df(self) -> None:
        df = self._make_df([])
        stats = compute_stats(df)
        assert stats.total_relationships == 0

    def test_basic_stats(self) -> None:
        rows = [
            {
                "relationship_id": "r1",
                "source_type": "observation",
                "source_id": "s1",
                "target_type": "observation",
                "target_id": "t1",
                "relationship_type": "similar",
                "similarity_score": 0.9,
                "confidence": 0.8,
                "metric": "cosine",
                "provider": "cosine",
                "model_fingerprint": "fp",
                "shared_entities": [],
                "shared_categories": [],
                "support_count": 0,
                "metadata": "{}",
                "version": "1.0",
                "created_at": "2026-01-01",
            },
            {
                "relationship_id": "r2",
                "source_type": "observation",
                "source_id": "s2",
                "target_type": "evidence",
                "target_id": "t2",
                "relationship_type": "similar",
                "similarity_score": 0.8,
                "confidence": 0.7,
                "metric": "cosine",
                "provider": "cosine",
                "model_fingerprint": "fp",
                "shared_entities": [],
                "shared_categories": [],
                "support_count": 0,
                "metadata": "{}",
                "version": "1.0",
                "created_at": "2026-01-01",
            },
        ]
        df = self._make_df(rows)
        stats = compute_stats(df, total_source_items=10)
        assert stats.total_relationships == 2
        assert stats.unique_source_ids == 2
        assert stats.unique_target_ids == 2
        assert stats.avg_similarity == pytest.approx(0.85, abs=1e-6)
        assert stats.source_type_counts["observation"] == 2
        assert stats.target_type_counts["observation"] == 1
        assert stats.target_type_counts["evidence"] == 1

    def test_density(self) -> None:
        rows = [
            {
                "relationship_id": "r1",
                "source_type": "observation",
                "source_id": "s1",
                "target_type": "observation",
                "target_id": "t1",
                "relationship_type": "similar",
                "similarity_score": 0.9,
                "confidence": 0.8,
                "metric": "cosine",
                "provider": "cosine",
                "model_fingerprint": "fp",
                "shared_entities": [],
                "shared_categories": [],
                "support_count": 0,
                "metadata": "{}",
                "version": "1.0",
                "created_at": "2026-01-01",
            },
        ]
        df = self._make_df(rows)
        stats = compute_stats(df, total_source_items=10)
        # 1 pair out of C(10,2) = 45 possible
        assert 0 < stats.density < 1
