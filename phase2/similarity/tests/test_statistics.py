"""Tests for RelationshipStatistics."""
from __future__ import annotations

import polars as pl
import pytest

from phase2.similarity.statistics import RelationshipStatistics, compute_relationship_statistics


def _make_df(rows: list[dict]) -> pl.DataFrame:
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


class TestComputeRelationshipStatistics:
    def test_empty_df(self) -> None:
        df = _make_df([])
        stats = compute_relationship_statistics(df)
        assert stats.total_relationships == 0
        assert stats.avg_similarity == 0.0
        assert stats.average_neighbors == 0.0

    def test_similarity_histogram(self) -> None:
        rows = [
            {
                "relationship_id": "r1", "source_type": "observation",
                "source_id": "s1", "target_type": "observation",
                "target_id": "t1", "relationship_type": "similar",
                "similarity_score": 0.7, "confidence": 0.6,
                "metric": "cosine", "provider": "cosine",
                "model_fingerprint": "fp", "shared_entities": [],
                "shared_categories": [], "support_count": 0,
                "metadata": "{}", "version": "1.0", "created_at": "now",
            },
            {
                "relationship_id": "r2", "source_type": "observation",
                "source_id": "s2", "target_type": "observation",
                "target_id": "t2", "relationship_type": "similar",
                "similarity_score": 0.9, "confidence": 0.8,
                "metric": "cosine", "provider": "cosine",
                "model_fingerprint": "fp", "shared_entities": [],
                "shared_categories": [], "support_count": 0,
                "metadata": "{}", "version": "1.0", "created_at": "now",
            },
        ]
        df = _make_df(rows)
        stats = compute_relationship_statistics(df, similarity_bins=5)
        assert len(stats.similarity_histogram_bins) == 6  # 5 + 1
        assert len(stats.similarity_histogram_counts) == 5
        assert sum(stats.similarity_histogram_counts) == 2

    def test_confidence_histogram(self) -> None:
        rows = [
            {
                "relationship_id": "r1", "source_type": "observation",
                "source_id": "s1", "target_type": "observation",
                "target_id": "t1", "relationship_type": "similar",
                "similarity_score": 0.8, "confidence": 0.6,
                "metric": "cosine", "provider": "cosine",
                "model_fingerprint": "fp", "shared_entities": [],
                "shared_categories": [], "support_count": 0,
                "metadata": "{}", "version": "1.0", "created_at": "now",
            },
            {
                "relationship_id": "r2", "source_type": "observation",
                "source_id": "s2", "target_type": "observation",
                "target_id": "t2", "relationship_type": "similar",
                "similarity_score": 0.8, "confidence": 0.8,
                "metric": "cosine", "provider": "cosine",
                "model_fingerprint": "fp", "shared_entities": [],
                "shared_categories": [], "support_count": 0,
                "metadata": "{}", "version": "1.0", "created_at": "now",
            },
        ]
        df = _make_df(rows)
        stats = compute_relationship_statistics(df, confidence_bins=4)
        assert len(stats.confidence_histogram_bins) == 5
        assert len(stats.confidence_histogram_counts) == 4

    def test_degree_distribution(self) -> None:
        rows = [
            {
                "relationship_id": f"r{i}", "source_type": "observation",
                "source_id": f"s{i}", "target_type": "observation",
                "target_id": f"t{i}", "relationship_type": "similar",
                "similarity_score": 0.8, "confidence": 0.7,
                "metric": "cosine", "provider": "cosine",
                "model_fingerprint": "fp", "shared_entities": [],
                "shared_categories": [], "support_count": 0,
                "metadata": "{}", "version": "1.0", "created_at": "now",
            }
            for i in range(5)
        ]
        df = _make_df(rows)
        stats = compute_relationship_statistics(df, total_source_items=10)
        assert stats.connected_nodes == 5
        assert stats.average_neighbors > 0
        assert stats.max_neighbors >= 1

    def test_isolated_nodes(self) -> None:
        rows = [
            {
                "relationship_id": "r1", "source_type": "observation",
                "source_id": "s1", "target_type": "observation",
                "target_id": "t1", "relationship_type": "similar",
                "similarity_score": 0.8, "confidence": 0.7,
                "metric": "cosine", "provider": "cosine",
                "model_fingerprint": "fp", "shared_entities": [],
                "shared_categories": [], "support_count": 0,
                "metadata": "{}", "version": "1.0", "created_at": "now",
            },
        ]
        df = _make_df(rows)
        stats = compute_relationship_statistics(df, total_source_items=10)
        # 10 total items, 1 connected = 9 isolated
        assert stats.isolated_nodes == 9

    def test_top_connected_sources(self) -> None:
        rows = [
            {
                "relationship_id": f"r{i}", "source_type": "observation",
                "source_id": "hub", "target_type": "observation",
                "target_id": f"t{i}", "relationship_type": "similar",
                "similarity_score": 0.8, "confidence": 0.7,
                "metric": "cosine", "provider": "cosine",
                "model_fingerprint": "fp", "shared_entities": [],
                "shared_categories": [], "support_count": 0,
                "metadata": "{}", "version": "1.0", "created_at": "now",
            }
            for i in range(5)
        ]
        df = _make_df(rows)
        stats = compute_relationship_statistics(df, top_k=3)
        assert len(stats.top_connected_sources) == 1  # only one source
        assert stats.top_connected_sources[0]["source_id"] == "hub"
        assert stats.top_connected_sources[0]["relationship_count"] == 5

    def test_relationship_type_counts(self) -> None:
        rows = [
            {
                "relationship_id": "r1", "source_type": "observation",
                "source_id": "s1", "target_type": "observation",
                "target_id": "t1", "relationship_type": "similar",
                "similarity_score": 0.8, "confidence": 0.7,
                "metric": "cosine", "provider": "cosine",
                "model_fingerprint": "fp", "shared_entities": [],
                "shared_categories": [], "support_count": 0,
                "metadata": "{}", "version": "1.0", "created_at": "now",
            },
            {
                "relationship_id": "r2", "source_type": "evidence",
                "source_id": "s1", "target_type": "observation",
                "target_id": "t1", "relationship_type": "duplicate",
                "similarity_score": 0.9, "confidence": 0.8,
                "metric": "cosine", "provider": "cosine",
                "model_fingerprint": "fp", "shared_entities": [],
                "shared_categories": [], "support_count": 0,
                "metadata": "{}", "version": "1.0", "created_at": "now",
            },
        ]
        df = _make_df(rows)
        stats = compute_relationship_statistics(df)
        assert stats.relationship_type_counts.get("similar") == 1
        assert stats.relationship_type_counts.get("duplicate") == 1


class TestRelationshipStatisticsDataclass:
    def test_defaults(self) -> None:
        stats = RelationshipStatistics()
        assert stats.total_relationships == 0
        assert stats.degree_histogram_bins == []
        assert stats.similarity_histogram_counts == []