"""Tests for ThresholdRecommender."""
from __future__ import annotations

import numpy as np
import pytest

from phase2.similarity.threshold import ThresholdRecommender


class TestThresholdRecommender:
    def test_recommend_with_scores(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.uniform(0.5, 0.95, size=1000).tolist()
        rec = ThresholdRecommender()
        result = rec.recommend(scores, configured_threshold=0.82)
        assert result.sample_size == 1000
        assert 0.5 <= result.min_similarity <= 0.95
        assert result.max_similarity >= result.min_similarity
        assert result.mean > 0
        assert result.median > 0
        assert result.std > 0

    def test_percentiles_ordered(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rec = ThresholdRecommender()
        result = rec.recommend(scores, configured_threshold=0.82)
        assert result.p50 <= result.p75 <= result.p90 <= result.p95 <= result.p99

    def test_candidate_thresholds(self) -> None:
        scores = [0.6, 0.7, 0.8, 0.9, 1.0]
        rec = ThresholdRecommender()
        result = rec.recommend(scores, configured_threshold=0.82, candidate_values=[0.65, 0.75, 0.85])
        for t in [0.65, 0.75, 0.85]:
            assert t in result.candidate_thresholds
            assert result.candidate_thresholds[t] >= 0

    def test_warning_below_range(self) -> None:
        scores = [0.7, 0.8, 0.9]
        rec = ThresholdRecommender()
        result = rec.recommend(scores, configured_threshold=0.10)
        assert result.threshold_warning is not None
        assert "below" in result.threshold_warning

    def test_warning_above_range(self) -> None:
        scores = [0.7, 0.8, 0.9]
        rec = ThresholdRecommender()
        result = rec.recommend(scores, configured_threshold=0.99)
        assert result.threshold_warning is not None
        assert "exceeds" in result.threshold_warning

    def test_no_warning_in_range(self) -> None:
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        rec = ThresholdRecommender()
        result = rec.recommend(scores, configured_threshold=0.75)
        assert result.threshold_warning is None

    def test_empty_scores(self) -> None:
        rec = ThresholdRecommender()
        result = rec.recommend([], configured_threshold=0.82)
        assert result.sample_size == 0
        assert result.threshold_warning == "no scores available for analysis"

    def test_recommended_range(self) -> None:
        scores = list(np.linspace(0.5, 0.95, 500))
        rec = ThresholdRecommender()
        result = rec.recommend(scores, configured_threshold=0.82)
        assert result.recommended_min <= result.recommended_max
        assert result.recommended_min >= 0.5
        assert result.recommended_max <= 0.95

    def test_all_scores_same(self) -> None:
        scores = [0.75] * 100
        rec = ThresholdRecommender()
        result = rec.recommend(scores, configured_threshold=0.75)
        assert result.min_similarity == 0.75
        assert result.max_similarity == 0.75
        assert result.std == 0.0
        assert result.p50 == result.p75 == result.p90 == result.p95 == result.p99


class TestThresholdRecommendationDataclass:
    def test_defaults(self) -> None:
        from phase2.similarity.threshold import ThresholdRecommendation
        rec = ThresholdRecommendation()
        assert rec.min_similarity == 0.0
        assert rec.max_similarity == 0.0
        assert rec.sample_size == 0
        assert rec.candidate_thresholds == {}
        assert rec.threshold_warning is None