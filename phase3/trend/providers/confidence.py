"""ConfidenceScoreProvider — scores trends based on data quality and snapshot count."""

from __future__ import annotations

from phase3.trend.providers.base import TrendScoreProvider
from phase3.trend.providers.registry import register_trend_score_provider
from phase3.trend.schema import TrendScoringBreakdown


@register_trend_score_provider(name="confidence", priority=70)
class ConfidenceScoreProvider(TrendScoreProvider):
    @property
    def name(self) -> str:
        return "confidence"

    @property
    def priority(self) -> int:
        return 70

    def score(self, candidate: dict, context: dict) -> TrendScoringBreakdown:
        snapshot_count = int(candidate.get("snapshot_count", 1))
        confidence = float(candidate.get("confidence", 0.0))
        obs_count = int(candidate.get("total_observations", 0))

        snap_factor = min(snapshot_count / 10.0, 1.0)
        obs_factor = min(obs_count / 1000.0, 1.0)
        combined = (confidence * 0.4) + (snap_factor * 0.3) + (obs_factor * 0.3)
        return TrendScoringBreakdown(confidence_score=round(combined, 4))
