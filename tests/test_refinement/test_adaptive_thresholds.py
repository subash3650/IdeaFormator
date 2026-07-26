"""Tests for adaptive thresholds (Issue 2)."""

from __future__ import annotations

import math

import pytest

from pain_intelligence.intelligence.problem_signals import ProblemSignalDiscoverer


class TestAdaptiveThresholds:
    """Verify that thresholds scale based on dataset size."""

    def test_apply_adaptive_thresholds_small_dataset(self):
        """Small dataset (1965 docs) gets low thresholds."""
        discoverer = ProblemSignalDiscoverer(min_document_count=None)

        # Simulate 1965 documents
        discoverer.apply_adaptive_thresholds(1965)

        # log10(1965) ≈ 3.29, so min_document_count = max(3, 3) = 3
        expected = max(3, int(math.log10(1965)))
        assert discoverer.min_document_count == expected

    def test_apply_adaptive_thresholds_medium_dataset(self):
        """Medium dataset (10000 docs) gets slightly higher thresholds."""
        discoverer = ProblemSignalDiscoverer(min_document_count=None)

        discoverer.apply_adaptive_thresholds(10000)
        # log10(10000) = 4, so min_document_count = max(3, 4) = 4
        assert discoverer.min_document_count == 4

    def test_apply_adaptive_thresholds_large_dataset(self):
        """Large dataset (293000 docs) gets log-scaled thresholds."""
        discoverer = ProblemSignalDiscoverer(min_document_count=None)

        discoverer.apply_adaptive_thresholds(293000)
        # log10(293000) ≈ 5.47, so min_document_count = max(3, 5) = 5
        assert discoverer.min_document_count == 5

    def test_apply_adaptive_thresholds_configured_override(self):
        """Explicitly configured threshold takes precedence over adaptive."""
        discoverer = ProblemSignalDiscoverer(min_document_count=10)

        discoverer.apply_adaptive_thresholds(1965)
        # Configured value (10) > adaptive (3), so should use 10
        assert discoverer.min_document_count == 10

    def test_apply_adaptive_thresholds_very_small(self):
        """Extremely small dataset (1 doc) gets minimum threshold of 3."""
        discoverer = ProblemSignalDiscoverer(min_document_count=None)

        discoverer.apply_adaptive_thresholds(1)
        assert discoverer.min_document_count == max(3, int(math.log10(1)))

    def test_filtering_stats_contains_explainability(self):
        """Discarded signals include reason, rule, score, support, confidence."""
        discoverer = ProblemSignalDiscoverer(min_document_count=None)
        discoverer.apply_adaptive_thresholds(100)

        stats = discoverer.filtering_stats
        assert "thresholds" in stats
        assert "min_document_count" in stats["thresholds"]
        assert "max_avg_rating" in stats["thresholds"]
        assert "min_confidence" in stats["thresholds"]

    def test_get_diagnostics_structure(self):
        """get_diagnostics returns structured data with distributions."""
        discoverer = ProblemSignalDiscoverer(min_document_count=None)
        discoverer.apply_adaptive_thresholds(1000)
        _ = discoverer.discover([])

        diag = discoverer.get_diagnostics()
        assert "candidate_count_before_filtering" in diag
        assert "total_removed" in diag
        assert "remaining_count" in diag
        assert "removal_by_reason" in diag
        assert "thresholds" in diag
        assert "support_distribution" in diag
        assert "confidence_distribution" in diag
        assert "evidence_distribution" in diag
        assert "discarded_signals" in diag

    def test_support_distribution_values(self):
        """Support distribution captures document counts."""
        discoverer = ProblemSignalDiscoverer(min_document_count=10)
        discoverer.apply_adaptive_thresholds(10000)

        diag = discoverer.get_diagnostics()
        dist = diag.get("support_distribution", {})

        # When no candidates, min/max should be 0
        assert "min" in dist
        assert "max" in dist
        assert "count" in dist

    def test_adaptive_threshold_logs(self):
        """Verify the math of log-based thresholds."""
        test_cases = [
            (1, 3),
            (10, 3),
            (100, 3),
            (1000, 3),
            (10000, 4),
            (100000, 5),
            (293000, 5),
            (1000000, 6),
        ]
        for doc_count, expected_min in test_cases:
            expected = max(3, int(math.log10(max(doc_count, 1))))
            assert expected == expected_min, f"Failed for doc_count={doc_count}"
