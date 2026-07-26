"""CompetitionScoreProvider — estimates competitive landscape density."""

from __future__ import annotations

from phase3.opportunity.providers.base import ScoringProvider
from phase3.opportunity.providers.registry import register_scoring_provider
from phase3.opportunity.schema import ScoringBreakdown


@register_scoring_provider(name="competition", priority=70)
class CompetitionScoreProvider(ScoringProvider):
    """Estimates competition from product/company density.

    Higher competition = lower score (inverse relationship).
    """

    @property
    def name(self) -> str:
        return "competition"

    @property
    def priority(self) -> int:
        return 70

    def score(self, candidate: dict, context: dict) -> ScoringBreakdown:
        product_count = candidate.get("product_count", 0)
        company_count = candidate.get("company_count", 0)
        total_products = context.get("total_products", 0)
        total_companies = context.get("total_companies", 0)

        total_entities = product_count + company_count
        total_all = max(total_products + total_companies, 1)
        density = total_entities / total_all

        novelty = 1.0 - min(1.0, density * 2)
        competition = 1.0 - novelty

        return ScoringBreakdown(
            competition=round(competition, 4),
            novelty=round(novelty, 4),
        )
