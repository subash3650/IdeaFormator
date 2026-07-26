"""GrowthScoreProvider — scores trends based on growth percentage."""

from __future__ import annotations

from phase3.trend.providers.base import TrendScoreProvider
from phase3.trend.providers.registry import register_trend_score_provider
from phase3.trend.schema import TrendScoringBreakdown


@register_trend_score_provider(name="growth", priority=100)
class GrowthScoreProvider(TrendScoreProvider):
    @property
    def name(self) -> str:
        return "growth"

    @property
    def priority(self) -> int:
        return 100

    def score(self, candidate: dict, context: dict) -> TrendScoringBreakdown:
        growth_pct = float(candidate.get("growth_pct", 0.0))
        base_score = min(abs(growth_pct) / 100.0, 1.0)
        return TrendScoringBreakdown(growth_score=round(base_score, 4))
