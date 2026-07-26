"""TrendScoreProvider — composite provider that combines other scores."""

from __future__ import annotations

from phase3.trend.providers.base import TrendScoreProvider
from phase3.trend.providers.registry import register_trend_score_provider
from phase3.trend.schema import TrendScoringBreakdown


@register_trend_score_provider(name="trend_score", priority=40)
class TrendScoreCompositeProvider(TrendScoreProvider):
    @property
    def name(self) -> str:
        return "trend_score"

    @property
    def priority(self) -> int:
        return 40

    def score(self, candidate: dict, context: dict) -> TrendScoringBreakdown:
        weights = context.get("score_weights", {})
        w_growth = float(weights.get("growth", 0.30))
        w_velocity = float(weights.get("velocity", 0.20))
        w_momentum = float(weights.get("momentum", 0.15))
        w_confidence = float(weights.get("confidence", 0.10))
        w_seasonality = float(weights.get("seasonality", 0.05))
        w_anomaly = float(weights.get("anomaly", 0.10))
        w_cross = float(weights.get("cross_platform", 0.10))

        gs = float(candidate.get("growth_score", 0.0))
        vs = float(candidate.get("velocity_score", 0.0))
        ms = float(candidate.get("momentum_score", 0.0))
        cs = float(candidate.get("confidence_score", 0.0))
        ss = float(candidate.get("seasonality_score", 0.0))
        ac = float(candidate.get("anomaly_score", 0.0))
        xs = float(candidate.get("cross_platform_score", 0.0))

        composite = (
            gs * w_growth + vs * w_velocity + ms * w_momentum + cs * w_confidence
            + ss * w_seasonality + ac * w_anomaly + xs * w_cross
        )
        return TrendScoringBreakdown(trend_score=round(min(composite, 1.0), 4))
