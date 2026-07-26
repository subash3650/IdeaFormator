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
from phase2.evaluation.schema import ClusterEvaluation, StageHealth


class ClusterEvaluator:
    def __init__(self, knowledge_dir: str | Path = "pain_intelligence/knowledge") -> None:
        self._phase2_dir = Path(knowledge_dir) / "assets" / "phase2"

    def evaluate(self, cluster_df: pl.DataFrame | None = None) -> ClusterEvaluation:
        if cluster_df is None:
            path = self._phase2_dir / "semantic_clusters.parquet"
            if not path.exists():
                return ClusterEvaluation()
            cluster_df = pl.read_parquet(str(path))

        result = ClusterEvaluation()
        total = cluster_df.height
        result.total_clusters = total

        if total == 0:
            result.health = StageHealth(score=0.0, weight=EvalConfig.weight("clusters"), warnings=["No clusters"])
            return result

        # Total members
        if column_exists(cluster_df, "member_count"):
            members = cluster_df["member_count"].to_list()
            result.total_members = sum(members) if members else 0

        # Cluster size distribution
        if column_exists(cluster_df, "member_count"):
            sizes = [float(v) for v in cluster_df["member_count"].to_list()]
            result.cluster_size_distribution = compute_distribution(sizes)

            result.singleton_count = sum(1 for s in sizes if s == 1.0)
            result.singleton_rate = round(safe_divide(result.singleton_count, total), 4)

        # Quality distribution
        if column_exists(cluster_df, "quality_score"):
            quals = [float(v) for v in cluster_df["quality_score"].drop_nulls().to_list()]
            if quals:
                result.quality_distribution = compute_distribution(quals)

            low_q = sum(
                1 for v in cluster_df["quality_score"].to_list()
                if v is not None and float(v) < EvalConfig.threshold("min_cluster_quality", 0.3)
            )
            result.low_quality_count = low_q
            result.low_quality_rate = round(safe_divide(low_q, total), 4)

        # Density distribution
        if column_exists(cluster_df, "density"):
            densities = [float(v) for v in cluster_df["density"].drop_nulls().to_list()]
            if densities:
                result.density_distribution = compute_distribution(densities)

        # Largest clusters
        if column_exists(cluster_df, "member_count"):
            sorted_df = cluster_df.sort("member_count", descending=True).head(5)
            result.largest_clusters = [
                {
                    "cluster_id": r.get("cluster_id", ""),
                    "member_count": r.get("member_count", 0),
                    "quality_score": r.get("quality_score", 0.0),
                }
                for r in sorted_df.iter_rows(named=True)
            ]

        # Cluster type distribution
        if column_exists(cluster_df, "cluster_type"):
            result.cluster_type_distribution = value_counts(cluster_df["cluster_type"])

        # Orphan concepts (nodes with member_count 1 clusters)
        result.orphan_concepts = result.singleton_count
        result.orphan_concept_rate = result.singleton_rate

        result.health = self._compute_health(result)
        return result

    def _compute_health(self, ev: ClusterEvaluation) -> StageHealth:
        score = 100.0
        warnings: list[str] = []

        if ev.total_clusters == 0:
            return StageHealth(score=0.0, weight=EvalConfig.weight("clusters"), warnings=["No clusters"])

        if ev.low_quality_rate > EvalConfig.threshold("max_low_quality_cluster_rate", 0.30):
            penalty = min(30, ev.low_quality_rate * 100)
            score -= penalty
            warnings.append(f"High low-quality cluster rate: {ev.low_quality_rate:.1%}")

        if ev.singleton_rate > EvalConfig.threshold("max_singleton_rate", 0.50):
            penalty = min(20, ev.singleton_rate * 50)
            score -= penalty
            warnings.append(f"High singleton rate: {ev.singleton_rate:.1%}")

        if ev.total_clusters > 0 and ev.total_members == 0:
            score -= 20
            warnings.append("Clusters exist but have no members")

        if ev.quality_distribution.mean < EvalConfig.threshold("min_cluster_quality", 0.3):
            score -= 15
            warnings.append(f"Low average cluster quality: {ev.quality_distribution.mean:.3f}")

        score = max(0.0, min(100.0, score))
        return StageHealth(
            score=round(score, 1),
            weight=EvalConfig.weight("clusters"),
            warnings=warnings,
        )
