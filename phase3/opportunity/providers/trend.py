"""TrendScoreProvider — estimates trend direction from temporal patterns."""

from __future__ import annotations

from phase3.opportunity.providers.base import ScoringProvider
from phase3.opportunity.providers.registry import register_scoring_provider
from phase3.opportunity.schema import ScoringBreakdown


@register_scoring_provider(name="trend", priority=80)
class TrendScoreProvider(ScoringProvider):
    """Estimates trend score from recency and growth of evidence."""

    @property
    def name(self) -> str:
        return "trend"

    @property
    def priority(self) -> int:
        return 80

    def score(self, candidate: dict, context: dict) -> ScoringBreakdown:
        evidence_cnt = candidate.get("evidence_count", 1)
        recent_ratio = candidate.get("recent_evidence_ratio", 0.5)
        growth = candidate.get("evidence_growth_rate", 0.0)

        trend = recent_ratio * 0.4 + min(1.0, evidence_cnt / 20) * 0.3 + min(1.0, max(0.0, growth)) * 0.3
        trend = min(1.0, max(0.0, trend))

        return ScoringBreakdown(
            trend=round(trend, 4),
        )
