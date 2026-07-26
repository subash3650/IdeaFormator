"""MomentumScoreProvider — scores trends based on momentum (weighted velocity trend)."""

from __future__ import annotations

from phase3.trend.providers.base import TrendScoreProvider
from phase3.trend.providers.registry import register_trend_score_provider
from phase3.trend.schema import TrendScoringBreakdown


@register_trend_score_provider(name="momentum", priority=80)
class MomentumScoreProvider(TrendScoreProvider):
    @property
    def name(self) -> str:
        return "momentum"

    @property
    def priority(self) -> int:
        return 80

    def score(self, candidate: dict, context: dict) -> TrendScoringBreakdown:
        momentum = float(candidate.get("momentum", 0.0))
        base_score = min(momentum, 1.0)
        return TrendScoringBreakdown(momentum_score=round(base_score, 4))
