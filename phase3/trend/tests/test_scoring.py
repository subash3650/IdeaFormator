"""Tests for TrendScorer."""

from __future__ import annotations

from pathlib import Path

from phase3.trend.config import TrendConfig
from phase3.trend.scoring import TrendScorer


class TestTrendScorer:
    def test_score_empty(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        scorer = TrendScorer(cfg)
        result = scorer.score([], {})
        assert result == []

    def test_score_single_candidate(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        scorer = TrendScorer(cfg)
        candidate = {
            "trend_id": "t1",
            "title": "Growing: Test",
            "subject_id": "p1",
            "subject_label": "Test Problem",
            "trend_type": "growing",
            "trend_direction": "up",
            "trend_subject": "problem",
            "snapshot_ids": ["s1", "s2"],
            "growth_pct": 50.0,
            "velocity": 10.0,
            "momentum": 0.5,
            "confidence": 0.8,
            "snapshot_count": 2,
            "total_observations": 100,
            "current_value": 100.0,
            "prior_value": 50.0,
            "metrics": {"growth_pct": 50.0, "snapshot_count": 2},
            "affected_products": [],
            "affected_companies": [],
            "affected_technologies": [],
            "affected_platforms": [],
            "growth_score": 0.0,
            "velocity_score": 0.0,
            "momentum_score": 0.0,
            "confidence_score": 0.0,
            "seasonality_score": 0.0,
            "anomaly_score": 0.0,
            "cross_platform_score": 0.0,
        }
        context = {"score_weights": {}, "total_snapshots": 2, "max_observations": 100}
        trends = scorer.score([candidate], context)
        assert len(trends) == 1
        t = trends[0]
        assert t.trend_id == "t1"
        assert t.trend_type.value == "growing"
        assert t.metrics.growth_pct == 50.0

    def test_providers_used(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        scorer = TrendScorer(cfg)
        assert len(scorer.providers_used) > 0
        assert "growth" in scorer.providers_used

    def test_score_with_enabled_providers(self, tmp_path: Path) -> None:
        cfg = TrendConfig(
            output_dir=tmp_path,
            enabled_scoring_providers=["growth", "velocity"],
        )
        scorer = TrendScorer(cfg)
        assert "growth" in scorer.providers_used
        assert "trend_score" not in scorer.providers_used

    def test_all_providers_contribute(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        scorer = TrendScorer(cfg)
        candidate = {
            "trend_id": "t1", "title": "Test", "subject_id": "p1",
            "subject_label": "Test", "trend_type": "growing",
            "trend_direction": "up", "trend_subject": "problem",
            "snapshot_ids": ["s1", "s2"],
            "growth_pct": 50.0, "velocity": 100.0, "momentum": 0.8,
            "confidence": 0.9, "snapshot_count": 5, "total_observations": 500,
            "current_value": 100.0, "prior_value": 50.0,
            "metrics": {"growth_pct": 50.0, "snapshot_count": 5},
            "affected_products": [], "affected_companies": [],
            "affected_technologies": [], "affected_platforms": [],
            "growth_score": 0.0, "velocity_score": 0.0, "momentum_score": 0.0,
            "confidence_score": 0.0, "seasonality_score": 0.0,
            "anomaly_score": 0.0, "cross_platform_score": 0.0,
        }
        context = {"score_weights": {}, "total_snapshots": 5, "max_observations": 500}
        trends = scorer.score([candidate], context)
        assert len(trends) == 1
        t = trends[0]
        assert t.scoring.growth_score > 0.0 or t.scoring.trend_score > 0.0
