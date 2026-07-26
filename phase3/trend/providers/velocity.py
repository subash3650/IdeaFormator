"""VelocityScoreProvider — scores trends based on velocity (per-day change rate)."""

from __future__ import annotations

from phase3.trend.providers.base import TrendScoreProvider
from phase3.trend.providers.registry import register_trend_score_provider
from phase3.trend.schema import TrendScoringBreakdown


@register_trend_score_provider(name="velocity", priority=90)
class VelocityScoreProvider(TrendScoreProvider):
    @property
    def name(self) -> str:
        return "velocity"

    @property
    def priority(self) -> int:
        return 90

    def score(self, candidate: dict, context: dict) -> TrendScoringBreakdown:
        velocity = float(candidate.get("velocity", 0.0))
        base_score = min(abs(velocity) / 1000.0, 1.0)
        return TrendScoringBreakdown(velocity_score=round(base_score, 4))
