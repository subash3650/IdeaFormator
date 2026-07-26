from __future__ import annotations

from pathlib import Path

import polars as pl

from phase2.evaluation.config import EvalConfig
from phase2.evaluation.metrics import (
    column_exists,
    compute_distribution,
    safe_divide,
    value_counts,
)
from phase2.evaluation.schema import EvidenceEvaluation, StageHealth


class EvidenceEvaluator:
    def __init__(self, knowledge_dir: str | Path = "pain_intelligence/knowledge") -> None:
        self._assets_dir = Path(knowledge_dir) / "assets"

    def evaluate(self, ev_df: pl.DataFrame | None = None, obs_count: int = 0) -> EvidenceEvaluation:
        if ev_df is None:
            path = self._assets_dir / "evidence.parquet"
            if not path.exists():
                return EvidenceEvaluation()
            ev_df = pl.read_parquet(str(path))

        result = EvidenceEvaluation()
        total = ev_df.height
        result.total_evidence = total
        result.total_observations = obs_count

        if total == 0:
            result.health = StageHealth(score=0.0, weight=EvalConfig.weight("evidence"), warnings=["No evidence"])
            return result

        # Compression ratio
        result.compression_ratio = round(safe_divide(obs_count or 1, total), 2)

        # Support distribution
        if column_exists(ev_df, "document_count"):
            vals = ev_df["document_count"].to_list()
            result.support_distribution = compute_distribution([float(v) for v in vals])

        # Category distribution
        if column_exists(ev_df, "category"):
            result.category_distribution = value_counts(ev_df["category"])

        # Entity type distribution
        if column_exists(ev_df, "entity_type"):
            result.entity_type_distribution = value_counts(ev_df["entity_type"])

        # Evidence confidence
        if column_exists(ev_df, "confidence"):
            conf_vals = ev_df["confidence"].drop_nulls().to_list()
            if conf_vals:
                result.evidence_confidence = compute_distribution([float(v) for v in conf_vals])

        # Avg observations per evidence
        if column_exists(ev_df, "observation_count"):
            obs_counts = ev_df["observation_count"].to_list()
            if obs_counts:
                result.avg_observations_per_evidence = round(
                    sum(obs_counts) / len(obs_counts), 2
                )

        result.health = self._compute_health(result)
        return result

    def _compute_health(self, ev: EvidenceEvaluation) -> StageHealth:
        score = 100.0
        warnings: list[str] = []

        if ev.total_evidence == 0:
            return StageHealth(score=0.0, weight=EvalConfig.weight("evidence"), warnings=["No evidence records"])

        if ev.compression_ratio < EvalConfig.threshold("min_compression_ratio", 1.0):
            score -= 20
            warnings.append(f"Low compression ratio: {ev.compression_ratio:.2f}")

        if ev.avg_observations_per_evidence < 2:
            score -= 15
            warnings.append(f"Low observations per evidence: {ev.avg_observations_per_evidence:.1f}")

        if ev.evidence_confidence.mean < EvalConfig.threshold("min_evidence_confidence", 0.5):
            score -= 10
            warnings.append(f"Low evidence confidence: {ev.evidence_confidence.mean:.3f}")

        if not ev.category_distribution or len(ev.category_distribution) < 2:
            score -= 10
            warnings.append("Limited category diversity")

        score = max(0.0, min(100.0, score))
        return StageHealth(
            score=round(score, 1),
            weight=EvalConfig.weight("evidence"),
            warnings=warnings,
        )
