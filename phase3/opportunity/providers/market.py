"""MarketScoreProvider — adjusts market coverage and feasibility scores."""

from __future__ import annotations

from phase3.opportunity.providers.base import ScoringProvider
from phase3.opportunity.providers.registry import register_scoring_provider
from phase3.opportunity.schema import ScoringBreakdown


@register_scoring_provider(name="market", priority=90)
class MarketScoreProvider(ScoringProvider):
    """Estimates market coverage from entity frequency and evidence spread."""

    @property
    def name(self) -> str:
        return "market"

    @property
    def priority(self) -> int:
        return 90

    def score(self, candidate: dict, context: dict) -> ScoringBreakdown:
        product_count = candidate.get("product_count", 0)
        company_count = candidate.get("company_count", 0)
        platform_count = candidate.get("platform_count", 1)
        total_platforms = context.get("total_platforms", 1)
        evidence_cnt = candidate.get("evidence_count", 1)

        combined = product_count + company_count
        market_coverage = min(1.0, combined / max(combined, 5) + platform_count / max(total_platforms, 1))
        market_coverage = min(1.0, market_coverage * 0.5)

        feasibility = min(1.0, evidence_cnt / max(evidence_cnt, 10) + 0.3)
        feasibility = min(1.0, feasibility)

        return ScoringBreakdown(
            market_coverage=round(market_coverage, 4),
            feasibility=round(feasibility, 4),
        )
