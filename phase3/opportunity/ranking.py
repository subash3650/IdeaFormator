"""OpportunityRanker — ranks and deduplicates scored opportunities."""

from __future__ import annotations

from typing import Any

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.providers.registry import register_ranking_provider
from phase3.opportunity.schema import Opportunity, OpportunityStatus, RankingStrategy


@register_ranking_provider(name="composite")
class WeightedCompositeRanker:
    """Default ranker: sort by opportunity_score descending, dedup, apply top-k."""

    @property
    def name(self) -> str:
        return "composite"

    @property
    def strategy(self) -> str:
        return RankingStrategy.COMPOSITE.value


class OpportunityRanker:
    """Ranks scored opportunities by configurable strategy."""

    def __init__(self, config: OpportunityConfig) -> None:
        self._config = config

    def rank(
        self,
        opportunities: list[Opportunity],
        context: dict[str, Any] | None = None,
    ) -> list[Opportunity]:
        if not opportunities:
            return []

        # Sort by strategy
        strategy = self._config.ranking_strategy
        if strategy == RankingStrategy.PAIN_SEVERITY:
            sorted_opps = sorted(opportunities, key=lambda o: -o.pain_severity)
        elif strategy == RankingStrategy.MARKET_SIZE:
            sorted_opps = sorted(opportunities, key=lambda o: -o.scoring_breakdown.market_coverage)
        elif strategy == RankingStrategy.FEASIBILITY:
            sorted_opps = sorted(opportunities, key=lambda o: -o.feasibility_score)
        elif strategy == RankingStrategy.CONFIDENCE:
            sorted_opps = sorted(opportunities, key=lambda o: -o.confidence.final_confidence)
        else:
            sorted_opps = sorted(opportunities, key=lambda o: -o.opportunity_score)

        # Deduplicate by evidence overlap
        deduped = self._deduplicate(sorted_opps)

        # Top-K
        top_k = min(self._config.top_k, len(deduped))
        ranked = deduped[:top_k]

        # Assign rank
        result: list[Opportunity] = []
        for i, opp in enumerate(ranked):
            result.append(opp.model_copy(update={
                "rank": i + 1,
                "status": OpportunityStatus.RANKED,
            }))

        return result

    def _deduplicate(self, opportunities: list[Opportunity]) -> list[Opportunity]:
        if not opportunities:
            return []
        threshold = self._config.dedup_similarity_threshold
        kept: list[Opportunity] = []
        for opp in opportunities:
            is_dup = False
            for existing in kept:
                overlap = self._evidence_overlap(opp, existing)
                if overlap >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(opp)
        return kept

    def _evidence_overlap(self, a: Opportunity, b: Opportunity) -> float:
        set_a = set(a.supporting_evidence)
        set_b = set(b.supporting_evidence)
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)
