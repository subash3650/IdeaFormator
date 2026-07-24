"""Tests for RelationshipConfidencePolicy."""

from __future__ import annotations

import numpy as np
import pytest

from phase2.similarity.confidence import RelationshipConfidencePolicy
from phase2.similarity.config import SimilarityEngineConfig


class TestRelationshipConfidencePolicy:
    def _make_policy(self) -> RelationshipConfidencePolicy:
        return RelationshipConfidencePolicy(SimilarityEngineConfig())

    def test_perfect_scores(self) -> None:
        policy = self._make_policy()
        conf = policy.compute(
            similarity_score=1.0,
            source_quality=1.0,
            target_quality=1.0,
            source_frequency=100,
            target_frequency=100,
            embedding_quality=1.0,
            metadata_completeness=1.0,
        )
        assert conf == pytest.approx(1.0, abs=1e-6)

    def test_zero_similarity(self) -> None:
        policy = self._make_policy()
        conf = policy.compute(similarity_score=0.0)
        assert conf == 0.0

    def test_default_factors(self) -> None:
        policy = self._make_policy()
        conf = policy.compute(similarity_score=0.8)
        assert 0.0 < conf < 0.8

    def test_batch_matches_individual(self) -> None:
        policy = self._make_policy()
        sims = [0.9, 0.8, 0.7]
        batch_result = policy.compute_batch(sims)
        for i, sim in enumerate(sims):
            individual = policy.compute(similarity_score=sim)
            assert batch_result[i] == pytest.approx(individual, abs=1e-6)

    def test_batch_with_frequencies(self) -> None:
        policy = self._make_policy()
        sims = [0.9, 0.8]
        sf = [10, 1]
        tf = [1, 10]
        result = policy.compute_batch(sims, source_frequencies=sf, target_frequencies=tf)
        assert len(result) == 2
        assert result[0] != result[1]

    def test_with_support_count(self) -> None:
        policy = self._make_policy()
        conf_high = policy.compute(similarity_score=0.8, support_count=100)
        conf_low = policy.compute(similarity_score=0.8, support_count=0)
        assert conf_high > conf_low

    def test_with_custom_weights(self) -> None:
        from phase2.similarity.confidence import ConfidenceFactors
        cfg = SimilarityEngineConfig()
        neutral = ConfidenceFactors(
            similarity_weight=1.0,
            source_quality_weight=0,
            target_quality_weight=0,
            source_frequency_weight=0,
            target_frequency_weight=0,
            embedding_quality_weight=0,
            support_count_weight=0,
            metadata_completeness_weight=0,
        )
        policy = RelationshipConfidencePolicy(cfg, weights=neutral)
        conf = policy.compute(similarity_score=0.75, source_frequency=1)
        # With only similarity active, confidence should be close to similarity
        assert conf == pytest.approx(0.75, abs=1e-6)

    def test_zero_weight_skips_factor(self) -> None:
        from phase2.similarity.confidence import ConfidenceFactors
        cfg = SimilarityEngineConfig()
        zero = ConfidenceFactors(
            similarity_weight=1.0,
            source_quality_weight=0, target_quality_weight=0,
            source_frequency_weight=0, target_frequency_weight=0,
            embedding_quality_weight=0, support_count_weight=0,
            metadata_completeness_weight=0,
        )
        policy = RelationshipConfidencePolicy(cfg, weights=zero)
        conf = policy.compute(similarity_score=0.8, source_quality=0.1)
        # source_quality of 0.1 should not affect result when weight is 0
        assert conf == pytest.approx(0.8, abs=1e-6)

    def test_support_count_batch(self) -> None:
        policy = self._make_policy()
        sims = [0.8, 0.8]
        sc = [100, 1]
        result = policy.compute_batch(sims, support_counts=sc)
        assert result[0] > result[1]
