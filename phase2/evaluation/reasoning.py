"""Reasoning stage evaluator for the pipeline evaluation framework."""

from __future__ import annotations

from pathlib import Path

from phase2.evaluation.schema import ReasoningEvaluation, StageHealth
from phase2.reasoning.schema import InferenceResult
from phase2.reasoning.store import ReasoningStore


class ReasoningEvaluator:
    def __init__(self, knowledge_dir: str | Path) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._store = ReasoningStore(self._knowledge_dir / "assets" / "phase2")

    def evaluate(self) -> ReasoningEvaluation:
        result = ReasoningEvaluation()
        inferences = self._store.load_inferences()
        chains = self._store.load_chains()
        root_causes = self._store.load_root_causes()
        evidence = self._store.load_evidence_aggregations()
        metadata = self._store.load_metadata()

        result.inference_count = len(inferences)
        result.chain_count = len(chains)
        result.root_cause_count = len(root_causes)
        result.evidence_aggregation_count = len(evidence)
        result.has_reasoning = result.inference_count > 0

        if metadata and metadata.rules_applied:
            result.rules_applied = metadata.rules_applied

        if inferences:
            result.avg_inference_confidence = round(
                sum(i.confidence for i in inferences) / len(inferences), 4
            )

        if root_causes:
            result.avg_root_cause_depth = round(
                sum(rc.path_length for rc in root_causes) / len(root_causes), 4
            )

        result.health = self._compute_health(result)
        return result

    def _compute_health(self, result: ReasoningEvaluation) -> StageHealth:
        score = 100.0
        warnings: list[str] = []
        metrics: dict[str, float] = {}

        metrics["inference_count"] = float(result.inference_count)
        metrics["avg_confidence"] = result.avg_inference_confidence
        metrics["root_cause_count"] = float(result.root_cause_count)
        metrics["evidence_count"] = float(result.evidence_aggregation_count)

        if not result.has_reasoning:
            score -= 40
            warnings.append("No reasoning results found — run 'reasoning reason' first")

        if result.inference_count < 5:
            score -= 20
            warnings.append(f"Low inference count ({result.inference_count})")

        if result.avg_inference_confidence < 0.3 and result.inference_count > 0:
            score -= 15
            warnings.append(f"Low average inference confidence ({result.avg_inference_confidence:.2f})")

        if result.root_cause_count == 0 and result.has_reasoning:
            score -= 10
            warnings.append("No root causes discovered")

        if result.evidence_aggregation_count == 0 and result.has_reasoning:
            score -= 10
            warnings.append("No evidence aggregations found")

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
