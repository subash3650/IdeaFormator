"""Opportunity stage evaluator for the pipeline evaluation framework."""

from __future__ import annotations

from pathlib import Path

from phase2.evaluation.schema import OpportunityEvaluation, StageHealth
from phase3.opportunity.store import OpportunityStore


class OpportunityEvaluator:
    """Evaluates the opportunity discovery stage."""

    def __init__(self, knowledge_dir: str | Path) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._store = OpportunityStore(self._knowledge_dir / "assets" / "phase3")

    def evaluate(self) -> OpportunityEvaluation:
        result = OpportunityEvaluation()
        opportunities = self._store.load_opportunities()
        metadata = self._store.load_metadata()

        result.total_opportunities = len(opportunities)
        result.has_opportunities = result.total_opportunities > 0

        if opportunities:
            scores = [o.opportunity_score for o in opportunities]
            result.avg_opportunity_score = round(sum(scores) / len(scores), 4)

            rec_dist: dict[str, int] = {}
            bm_dist: dict[str, int] = {}
            strong_pursue = 0
            worth_exploring = 0
            for o in opportunities:
                rec_dist[o.recommendation_type.value] = rec_dist.get(o.recommendation_type.value, 0) + 1
                bm_dist[o.suggested_business_model.value] = bm_dist.get(o.suggested_business_model.value, 0) + 1
                if o.recommendation_type.value == "strong_pursue":
                    strong_pursue += 1
                elif o.recommendation_type.value == "worth_exploring":
                    worth_exploring += 1

            result.strong_pursue_count = strong_pursue
            result.worth_exploring_count = worth_exploring
            result.recommendation_distribution = rec_dist
            result.business_model_distribution = bm_dist

        result.health = self._compute_health(result)
        return result

    def _compute_health(self, result: OpportunityEvaluation) -> StageHealth:
        score = 100.0
        warnings: list[str] = []
        metrics: dict[str, float] = {}

        metrics["total_opportunities"] = float(result.total_opportunities)
        metrics["avg_score"] = result.avg_opportunity_score
        metrics["strong_pursue_count"] = float(result.strong_pursue_count)
        metrics["worth_exploring_count"] = float(result.worth_exploring_count)

        if not result.has_opportunities:
            score -= 50
            warnings.append("No opportunities found — run 'opportunity discover' first")
        elif result.total_opportunities < 3:
            score -= 20
            warnings.append(f"Very few opportunities ({result.total_opportunities})")
        elif result.total_opportunities < 10:
            score -= 10
            warnings.append(f"Low opportunity count ({result.total_opportunities})")

        if result.strong_pursue_count == 0 and result.has_opportunities:
            score -= 10
            warnings.append("No 'strong pursue' opportunities identified")

        if result.avg_opportunity_score < 0.3 and result.has_opportunities:
            score -= 10
            warnings.append(f"Low average opportunity score ({result.avg_opportunity_score:.2f})")

        score = max(0.0, min(100.0, score))
        degradation = 100.0 - score

        return StageHealth(
            score=round(score, 1),
            max_score=100.0,
            weight=1.0,
            metrics=metrics,
            warnings=warnings,
            degradation_pct=round(degradation, 1),
        )
