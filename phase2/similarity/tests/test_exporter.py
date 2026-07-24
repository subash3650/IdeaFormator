"""Tests for manifest and report generation.

Extended to cover JSON report and threshold recommendations in text report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from phase2.similarity.exporter import generate_quality_report, write_json_report, write_manifest
from phase2.similarity.metrics import RelationshipStats
from phase2.similarity.schema import RelationshipManifest
from phase2.similarity.threshold import ThresholdRecommender


class TestWriteManifest:
    def test_writes_json(self, tmp_path: Path) -> None:
        manifest = RelationshipManifest(
            embedding_model="all-MiniLM-L6-v2",
            embedding_fingerprint="sentence_transformers/all-MiniLM-L6-v2@384d",
            metric="cosine",
            threshold=0.82,
            record_count=100,
            generated_at="2026-01-01T00:00:00",
            elapsed_seconds=5.0,
        )
        path = write_manifest(manifest, tmp_path)
        assert path.exists()
        assert path.name == "similarity_manifest.json"
        content = path.read_text(encoding="utf-8")
        assert "record_count" in content


class TestGenerateQualityReport:
    def test_generates_report(self, tmp_path: Path) -> None:
        stats = RelationshipStats(
            total_relationships=100,
            unique_source_ids=50,
            unique_target_ids=50,
            avg_similarity=0.85,
            avg_confidence=0.78,
            density=0.01,
        )
        path = generate_quality_report(stats, tmp_path, elapsed_seconds=2.5)
        assert path.exists()
        assert path.name == "similarity_report.txt"
        content = path.read_text(encoding="utf-8")
        assert "Semantic Relationship Quality Report" in content
        assert "100" in content

    def test_empty_stats(self, tmp_path: Path) -> None:
        stats = RelationshipStats()
        path = generate_quality_report(stats, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "FAIL" in content

    def test_with_threshold_recommendations(self, tmp_path: Path) -> None:
        stats = RelationshipStats(total_relationships=50)
        scores = [0.7, 0.8, 0.9]
        rec = ThresholdRecommender().recommend(scores, configured_threshold=0.82)
        path = generate_quality_report(
            stats, tmp_path, threshold_rec=rec, configured_threshold=0.82
        )
        content = path.read_text(encoding="utf-8")
        assert "Threshold Recommendations" in content
        assert "P50" in content
        assert "P95" in content

    def test_with_filter_counts(self, tmp_path: Path) -> None:
        stats = RelationshipStats(total_relationships=50)
        filter_counts = {
            "SelfSimilarityFilter": 5,
            "ThresholdFilter": 10,
            "total_survived": 50,
        }
        path = generate_quality_report(stats, tmp_path, filter_counts=filter_counts)
        content = path.read_text(encoding="utf-8")
        assert "Filter Pipeline" in content


class TestWriteJsonReport:
    def test_writes_json_report(self, tmp_path: Path) -> None:
        stats = RelationshipStats(
            total_relationships=100,
            unique_source_ids=50,
            unique_target_ids=40,
            unique_pair_ids=100,
            avg_similarity=0.85,
            avg_confidence=0.78,
            std_similarity=0.05,
            std_confidence=0.04,
            min_similarity=0.7,
            max_similarity=0.99,
            density=0.001,
            source_type_counts={"observation": 100},
            target_type_counts={"observation": 100},
        )
        path = write_json_report(stats, tmp_path, elapsed_seconds=3.0)
        assert path.exists()
        assert path.name == "similarity_report.json"
        content = path.read_text(encoding="utf-8")
        assert "similarity" in content
        assert "confidence" in content
        assert "density" in content

    def test_json_report_with_threshold_rec(self, tmp_path: Path) -> None:
        stats = RelationshipStats(total_relationships=50)
        scores = list(np.linspace(0.6, 0.95, 200))
        rec = ThresholdRecommender().recommend(scores, configured_threshold=0.82)
        path = write_json_report(stats, tmp_path, threshold_rec=rec)
        content = path.read_text(encoding="utf-8")
        assert "threshold_recommendations" in content
        assert "percentiles" in content
        assert "candidate_thresholds" in content

    def test_json_report_with_filter_counts(self, tmp_path: Path) -> None:
        stats = RelationshipStats(total_relationships=50)
        filter_counts = {"SelfSimilarityFilter": 5, "total_survived": 50}
        path = write_json_report(stats, tmp_path, filter_counts=filter_counts)
        content = path.read_text(encoding="utf-8")
        assert "filter_statistics" in content

    def test_json_report_with_dataframe(self, tmp_path: Path) -> None:
        stats = RelationshipStats(
            total_relationships=3,
            unique_source_ids=3,
            unique_target_ids=3,
            unique_pair_ids=3,
            avg_similarity=0.8,
        )
        data = {
            "relationship_id": ["r1", "r2", "r3"],
            "source_type": ["observation"] * 3,
            "source_id": ["s1", "s2", "s3"],
            "target_type": ["observation"] * 3,
            "target_id": ["t1", "t2", "t3"],
            "relationship_type": ["similar"] * 3,
            "similarity_score": [0.8, 0.85, 0.9],
            "confidence": [0.7, 0.75, 0.8],
            "metric": ["cosine"] * 3,
            "provider": ["cosine"] * 3,
            "model_fingerprint": ["fp"] * 3,
            "shared_entities": [[]] * 3,
            "shared_categories": [[]] * 3,
            "support_count": [0] * 3,
            "metadata": ["{}"] * 3,
            "version": ["1.0"] * 3,
            "created_at": ["now"] * 3,
        }
        df = pl.DataFrame(data)
        path = write_json_report(stats, tmp_path, df=df)
        content = path.read_text(encoding="utf-8")
        assert "degree_distribution" in content
        assert "top_connected_sources" in content
        assert "similarity_histogram" in content

    def test_json_report_empty_stats(self, tmp_path: Path) -> None:
        stats = RelationshipStats()
        path = write_json_report(stats, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "report_type" in content