"""Data-driven threshold recommendation based on similarity score distribution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class ThresholdRecommendation:
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    sample_size: int = 0
    recommended_min: float = 0.0
    recommended_max: float = 0.0
    candidate_thresholds: dict[float, int] = field(default_factory=dict)
    configured_threshold: float = 0.82
    threshold_warning: str | None = None


class ThresholdRecommender:
    """Analyzes similarity score distribution and recommends thresholds.

    Examines the distribution of raw (pre-filter) similarity scores
    and produces statistics plus estimated relationship counts at
    common threshold levels.  Does NOT modify the configured threshold.
    """

    def recommend(
        self,
        scores: Sequence[float],
        configured_threshold: float = 0.82,
        candidate_values: list[float] | None = None,
    ) -> ThresholdRecommendation:
        """Generate threshold recommendations from a list of similarity scores.

        Args:
            scores: Raw pre-filter similarity scores.
            configured_threshold: The user-configured threshold.
            candidate_values: Threshold levels to estimate counts for.

        Returns:
            A ThresholdRecommendation dataclass.
        """
        arr = np.asarray(scores, dtype=np.float64)
        n = len(arr)

        if n == 0:
            rec = ThresholdRecommendation(configured_threshold=configured_threshold)
            rec.threshold_warning = "no scores available for analysis"
            return rec

        rec = ThresholdRecommendation(
            min_similarity=float(arr.min()),
            max_similarity=float(arr.max()),
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            std=float(arr.std()),
            p50=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            sample_size=n,
            configured_threshold=configured_threshold,
        )

        # Candidate thresholds
        if candidate_values is None:
            candidate_values = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        counts: dict[float, int] = {}
        for t in candidate_values:
            counts[t] = int((arr >= t).sum())
        rec.candidate_thresholds = counts

        # Recommended range: p50 to p95
        rec.recommended_min = float(np.percentile(arr, 50))
        rec.recommended_max = float(np.percentile(arr, 95))

        # Warning if configured threshold is outside recommended range
        if configured_threshold < rec.recommended_min:
            rec.threshold_warning = (
                f"configured threshold {configured_threshold:.2f} is below the "
                f"50th percentile ({rec.recommended_min:.2f}); "
                "relationships may be too permissive"
            )
        elif configured_threshold > rec.recommended_max:
            rec.threshold_warning = (
                f"configured threshold {configured_threshold:.2f} exceeds the "
                f"95th percentile ({rec.recommended_max:.2f}); "
                "very few relationships may be generated"
            )

        return rec