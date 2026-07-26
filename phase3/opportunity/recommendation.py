"""RecommendationEngine — assigns recommendation types and business models."""

from __future__ import annotations

from typing import Any

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.providers.registry import (
    available_business_model_providers,
    create_business_model_provider,
)
from phase3.opportunity.schema import (
    Opportunity,
    OpportunityStatus,
    OpportunityType,
    RecommendationType,
)


class RecommendationEngine:
    """Assigns recommendation types and best-fit business models to opportunities."""

    def __init__(self, config: OpportunityConfig) -> None:
        self._config = config
        self._model_providers: list = []
        for name in available_business_model_providers():
            if name in config.enabled_business_model_providers:
                self._model_providers.append(create_business_model_provider(name))

    @property
    def providers_used(self) -> list[str]:
        return [p.name for p in self._model_providers]

    def recommend(self, opportunities: list[Opportunity], context: dict[str, Any] | None = None) -> list[Opportunity]:
        if not opportunities:
            return []

        result: list[Opportunity] = []
        for opp in opportunities:
            rec_type = self._assign_recommendation(opp)
            biz_model = self._select_business_model(opp, context or {})
            solution = self._generate_solution(opp, biz_model)

            result.append(opp.model_copy(update={
                "recommendation_type": rec_type,
                "suggested_business_model": biz_model,
                "suggested_solution": solution,
                "status": OpportunityStatus.RECOMMENDED,
            }))

        return result

    def _assign_recommendation(self, opp: Opportunity) -> RecommendationType:
        score = opp.opportunity_score
        if score >= self._config.strong_pursue_threshold:
            return RecommendationType.STRONG_PURSUE
        if score >= self._config.worth_exploring_threshold:
            return RecommendationType.WORTH_EXPLORING
        if score >= self._config.niche_potential_threshold:
            return RecommendationType.NICHE_POTENTIAL
        if score >= self._config.monitor_threshold:
            return RecommendationType.MONITOR
        return RecommendationType.INSUFFICIENT_DATA

    def _select_business_model(self, opp: Opportunity, context: dict) -> OpportunityType:
        best_model = OpportunityType.SAAS
        best_score = 0.0

        for provider in self._model_providers:
            model_type, score = provider.evaluate(opp, context)
            if score > best_score:
                best_score = score
                best_model = model_type

        return best_model

    def _generate_solution(self, opp: Opportunity, biz_model: OpportunityType) -> str:
        product_count = len(opp.affected_products)
        company_count = len(opp.affected_companies)
        evidence_count = len(opp.supporting_evidence)
        platform_count = len(set(opp.kg_node_ids)) if opp.kg_node_ids else 1

        model_display = biz_model.value.replace("_", " ").title()
        products_str = ", ".join(opp.affected_products[:3])
        if len(opp.affected_products) > 3:
            products_str += f", and {len(opp.affected_products) - 3} others"

        if products_str:
            solution = (
                f"A {model_display} that addresses '{opp.root_problem}' for users of {products_str}, "
                f"validated by {evidence_count} evidence sources across {platform_count} platforms "
                f"with a confidence score of {opp.confidence.final_confidence:.0%}."
            )
        else:
            solution = (
                f"A {model_display} that addresses '{opp.root_problem}', "
                f"validated by {evidence_count} evidence sources across {platform_count} platforms "
                f"with a confidence score of {opp.confidence.final_confidence:.0%}."
            )
        return solution[:500]
