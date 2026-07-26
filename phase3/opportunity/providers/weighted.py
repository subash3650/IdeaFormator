"""WeightedScoreProvider — default composite scoring provider."""

from __future__ import annotations

from phase3.opportunity.providers.base import ScoringProvider
from phase3.opportunity.providers.registry import register_scoring_provider
from phase3.opportunity.schema import ScoreWeights, ScoringBreakdown


@register_scoring_provider(name="weighted", priority=100)
class WeightedScoreProvider(ScoringProvider):
    """Default composite scorer using configurable ScoreWeights."""

    def __init__(self) -> None:
        self._weights = ScoreWeights().normalize()

    @property
    def name(self) -> str:
        return "weighted"

    @property
    def priority(self) -> int:
        return 100

    def score(self, candidate: dict, context: dict) -> ScoringBreakdown:
        """Compute weighted scoring breakdown from candidate fields."""
        severity = candidate.get("pain_severity", 0.5)
        frequency = candidate.get("frequency_score", 0.5)
        trend = candidate.get("trend_score", 0.5)
        evidence_cnt = candidate.get("evidence_count", 1)
        max_evidence = context.get("max_evidence_count", 1)
        reasoning_conf = candidate.get("reasoning_confidence", 0.5)
        cluster_density = candidate.get("cluster_density", 0.5)
        platform_count = candidate.get("platform_count", 1)
        total_platforms = context.get("total_platforms", 1)
        product_count = candidate.get("product_count", 0)
        max_products = context.get("max_product_count", 1)
        competition_val = candidate.get("competition_score", 0.5)
        feasibility = candidate.get("feasibility_score", 0.5)
        novelty = candidate.get("novelty_score", 0.5)

        ev_score = min(1.0, evidence_cnt / max(max_evidence, 1))
        cp_score = min(1.0, platform_count / max(total_platforms, 1))
        mc_score = min(1.0, product_count / max(max_products, 1))

        return ScoringBreakdown(
            pain_severity=min(1.0, max(0.0, severity)),
            frequency=min(1.0, max(0.0, frequency)),
            trend=min(1.0, max(0.0, trend)),
            evidence_count=ev_score,
            reasoning_confidence=min(1.0, max(0.0, reasoning_conf)),
            cluster_density=min(1.0, max(0.0, cluster_density)),
            cross_platform=cp_score,
            market_coverage=mc_score,
            competition=min(1.0, max(0.0, competition_val)),
            feasibility=min(1.0, max(0.0, feasibility)),
            novelty=min(1.0, max(0.0, novelty)),
        )
