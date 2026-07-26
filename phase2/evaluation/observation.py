from __future__ import annotations

from pathlib import Path

import polars as pl

from phase2.evaluation.config import EvalConfig
from phase2.evaluation.metrics import (
    column_exists,
    compute_distribution,
    entropy,
    safe_divide,
    uniqueness_ratio,
    value_counts,
)
from phase2.evaluation.schema import ObservationEvaluation, StageHealth


class ObservationEvaluator:
    def __init__(self, knowledge_dir: str | Path = "pain_intelligence/knowledge") -> None:
        self._assets_dir = Path(knowledge_dir) / "assets"
        self._doc_dir = Path(knowledge_dir) / "processed"

    def evaluate(self, obs_df: pl.DataFrame | None = None, doc_count: int = 0) -> ObservationEvaluation:
        if obs_df is None:
            path = self._assets_dir / "observations.parquet"
            if not path.exists():
                return ObservationEvaluation()
            obs_df = pl.read_parquet(str(path))

        result = ObservationEvaluation()
        total = obs_df.height
        result.total_observations = total
        result.total_documents = doc_count or self._estimate_doc_count(obs_df)

        if total == 0:
            result.health = StageHealth(score=0.0, weight=EvalConfig.weight("observations"), warnings=["No observations"])
            return result

        # Observations per document
        if column_exists(obs_df, "document_id"):
            doc_counts = obs_df["document_id"].value_counts()
            cnt_col = [c for c in doc_counts.columns if c != "document_id"][0]
            result.observations_per_document = compute_distribution(
                [float(r[cnt_col]) for r in doc_counts.iter_rows(named=True)]
            )

        # Type distribution
        if column_exists(obs_df, "type"):
            result.type_distribution = value_counts(obs_df["type"])

        # Extractor contribution
        if column_exists(obs_df, "extractor"):
            result.extractor_contribution = value_counts(obs_df["extractor"])
            result.extractor_contribution_pct = {
                k: round(v / total * 100, 2) for k, v in result.extractor_contribution.items()
            }

        # Entity precision (observations with entity assigned)
        if column_exists(obs_df, "entity"):
            non_null_entity = obs_df["entity"].drop_nulls().len()
            result.entity_precision = round(safe_divide(non_null_entity, total), 4)
            result.entity_coverage = round(
                uniqueness_ratio(obs_df["entity"].drop_nulls()), 4
            ) if non_null_entity > 0 else 0.0

        # Keyword/phrase/pattern diversity
        if column_exists(obs_df, "type") and column_exists(obs_df, "value"):
            type_groups = obs_df.group_by("type").agg(pl.col("value"))
            for t_name in ["keyword", "phrase", "pattern"]:
                subset = obs_df.filter(pl.col("type") == t_name)
                if subset.height > 0 and column_exists(subset, "value"):
                    diversity = uniqueness_ratio(subset["value"])
                    if t_name == "keyword":
                        result.keyword_diversity = round(diversity, 4)
                    elif t_name == "phrase":
                        result.phrase_diversity = round(diversity, 4)
                    elif t_name == "pattern":
                        result.pattern_diversity = round(diversity, 4)

        # Canonicalization success
        if column_exists(obs_df, "canonical_value"):
            cv_count = obs_df["canonical_value"].drop_nulls().len()
            result.canonicalization_success_rate = round(safe_divide(cv_count, total), 4)

        # Knowledge enrichment coverage
        if column_exists(obs_df, "entity"):
            matched = obs_df.filter(pl.col("entity").is_not_null()).height
            result.knowledge_enrichment_coverage = round(safe_divide(matched, total), 4)

        result.health = self._compute_health(result)
        return result

    def _estimate_doc_count(self, obs_df: pl.DataFrame) -> int:
        if "document_id" in obs_df.columns:
            return obs_df["document_id"].n_unique()
        return 0

    def _compute_health(self, ev: ObservationEvaluation) -> StageHealth:
        score = 100.0
        warnings: list[str] = []

        if ev.total_observations == 0:
            return StageHealth(score=0.0, weight=EvalConfig.weight("observations"), warnings=["No observations"])

        if ev.total_documents > 0:
            obs_per_doc = safe_divide(ev.total_observations, ev.total_documents)
            if obs_per_doc < 0.1:
                score -= 20
                warnings.append(f"Very low observation extraction: {obs_per_doc:.2f} per document")

        if ev.knowledge_enrichment_coverage < 0.1:
            score -= 15
            warnings.append(f"Low enrichment coverage: {ev.knowledge_enrichment_coverage:.1%}")

        if ev.canonicalization_success_rate < 0.1:
            score -= 10
            warnings.append(f"Low canonicalization: {ev.canonicalization_success_rate:.1%}")

        extractor_count = len(ev.extractor_contribution)
        if extractor_count < 2:
            score -= 10
            warnings.append(f"Only {extractor_count} extractor(s) contributed")

        score = max(0.0, min(100.0, score))
        return StageHealth(
            score=round(score, 1),
            weight=EvalConfig.weight("observations"),
            warnings=warnings,
        )
