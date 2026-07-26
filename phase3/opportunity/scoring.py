"""OpportunityScorer — orchestrates provider-based scoring of candidates."""

from __future__ import annotations

from typing import Any

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.providers.registry import (
    create_scoring_provider,
    sorted_scoring_providers,
)
from phase3.opportunity.schema import (
    ConfidenceBreakdown,
    MarketSize,
    Opportunity,
    OpportunityMetadata,
    OpportunityStatus,
    RecommendationType,
    ScoreWeights,
    ScoringBreakdown,
)


class OpportunityScorer:
    """Orchestrates all scoring providers to convert candidates into Opportunities."""

    def __init__(self, config: OpportunityConfig) -> None:
        self._config = config
        self._weights = config.score_weights.normalize()
        self._providers: list = []
        for name in sorted_scoring_providers():
            if name in config.enabled_scoring_providers:
                self._providers.append(create_scoring_provider(name))

    @property
    def providers_used(self) -> list[str]:
        return [p.name for p in self._providers]

    def score(
        self,
        candidates: list[dict],
        context: dict[str, Any],
        run_id: str,
    ) -> list[Opportunity]:
        if not candidates:
            return []

        opportunities: list[Opportunity] = []

        for cand in candidates:
            breakdown = ScoringBreakdown()

            # Run all scoring providers in priority order
            for provider in self._providers:
                partial = provider.score(cand, context)
                breakdown = self._merge_breakdowns(breakdown, partial)

            # Compute final opportunity score from weighted breakdown
            opportunity_score = self._compute_composite(breakdown)

            # Compute confidence breakdown
            confidence = self._compute_confidence(cand, breakdown)

            # Determine market size
            market_size = self._estimate_market_size(cand, breakdown)

            opp = Opportunity(
                opportunity_id=self._deterministic_id(cand, run_id),
                title=cand.get("title", ""),
                summary=cand.get("summary", ""),
                root_problem=cand.get("root_problem", ""),
                supporting_evidence=cand.get("evidence_ids", []),
                reasoning_chain_ids=cand.get("reasoning_chain_ids", []),
                cluster_ids=cand.get("cluster_ids", []),
                kg_node_ids=cand.get("kg_node_ids", []),
                affected_products=cand.get("affected_products", []),
                affected_companies=cand.get("affected_companies", []),
                affected_technologies=cand.get("affected_technologies", []),
                estimated_market_size=market_size,
                pain_severity=breakdown.pain_severity,
                frequency_score=breakdown.frequency,
                trend_score=breakdown.trend,
                competition_score=breakdown.competition,
                feasibility_score=breakdown.feasibility,
                opportunity_score=round(opportunity_score, 4),
                scoring_breakdown=breakdown,
                confidence=confidence,
                recommendation_type=RecommendationType.INSUFFICIENT_DATA,
                status=OpportunityStatus.SCORED,
                pipeline_version=self._config.version,
                schema_version="1.0",
            )
            opportunities.append(opp)

        return opportunities

    def _merge_breakdowns(self, base: ScoringBreakdown, partial: ScoringBreakdown) -> ScoringBreakdown:
        data = base.model_dump()
        for key, val in partial.model_dump().items():
            if isinstance(val, float) and val != 0.0:
                data[key] = min(1.0, max(0.0, val))
        return ScoringBreakdown(**data)

    def _compute_composite(self, breakdown: ScoringBreakdown) -> float:
        w = self._weights
        return (
            breakdown.pain_severity * w.pain_severity
            + breakdown.frequency * w.frequency
            + breakdown.trend * w.trend
            + breakdown.evidence_count * w.evidence_count
            + breakdown.reasoning_confidence * w.reasoning_confidence
            + breakdown.cluster_density * w.cluster_density
            + breakdown.cross_platform * w.cross_platform
            + breakdown.market_coverage * w.market_coverage
            + breakdown.competition * w.competition
            + breakdown.feasibility * w.feasibility
            + breakdown.novelty * w.novelty
        )

    def _compute_confidence(self, cand: dict, breakdown: ScoringBreakdown) -> ConfidenceBreakdown:
        cfg = self._config
        rc = cand.get("reasoning_confidence", 0.5)
        ev = breakdown.evidence_count
        gc = cand.get("cluster_density", 0.5)
        mc = breakdown.market_coverage

        final = (
            rc * cfg.reasoning_confidence_weight
            + ev * cfg.evidence_confidence_weight
            + gc * cfg.graph_confidence_weight
            + mc * cfg.market_confidence_weight
        )
        return ConfidenceBreakdown(
            reasoning_confidence=round(min(1.0, rc), 4),
            evidence_confidence=round(min(1.0, ev), 4),
            graph_confidence=round(min(1.0, gc), 4),
            market_confidence=round(min(1.0, mc), 4),
            final_confidence=round(min(1.0, max(0.0, final)), 4),
            computation_method=cfg.confidence_method,
        )

    def _estimate_market_size(self, cand: dict, breakdown: ScoringBreakdown) -> MarketSize:
        product_count = cand.get("product_count", 0)
        company_count = cand.get("company_count", 0)
        platform_count = cand.get("platform_count", 1)
        evidence_cnt = cand.get("evidence_count", 0)
        combined = product_count + company_count + platform_count + evidence_cnt

        if combined >= 50:
            return MarketSize.LARGE
        if combined >= 20:
            return MarketSize.MEDIUM
        if combined >= 5:
            return MarketSize.SMALL
        return MarketSize.UNKNOWN

    def _deterministic_id(self, cand: dict, run_id: str) -> str:
        import hashlib
        raw = f"{run_id}:{cand.get('root_problem', '')}:{cand.get('title', '')}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
