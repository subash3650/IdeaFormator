from __future__ import annotations

from pathlib import Path

import polars as pl
import numpy as np

from phase2.evaluation.config import EvalConfig
from phase2.evaluation.metrics import (
    column_exists,
    compute_distribution,
    safe_divide,
    value_counts,
)
from phase2.evaluation.schema import DistributionStats, RelationshipEvaluation, StageHealth


class RelationshipEvaluator:
    def __init__(self, knowledge_dir: str | Path = "pain_intelligence/knowledge") -> None:
        self._phase2_dir = Path(knowledge_dir) / "assets" / "phase2"

    def evaluate(self, rel_df: pl.DataFrame | None = None) -> RelationshipEvaluation:
        if rel_df is None:
            path = self._phase2_dir / "semantic_relationships.parquet"
            if not path.exists():
                return RelationshipEvaluation()
            rel_df = pl.read_parquet(str(path))

        result = RelationshipEvaluation()
        total = rel_df.height
        result.total_relationships = total

        if total == 0:
            result.health = StageHealth(score=0.0, weight=EvalConfig.weight("relationships"), warnings=["No relationships"])
            return result

        # Average similarity
        if column_exists(rel_df, "similarity_score"):
            sims = rel_df["similarity_score"].drop_nulls().to_list()
            if sims:
                result.average_similarity = round(sum(sims) / len(sims), 6)
                result.similarity_distribution = compute_distribution([float(v) for v in sims])

        # Relationship type distribution
        if column_exists(rel_df, "relationship_type"):
            result.relationship_type_distribution = value_counts(rel_df["relationship_type"])

        # Confidence distribution
        if column_exists(rel_df, "confidence"):
            cv = rel_df["confidence"].drop_nulls().to_list()
            if cv:
                result.confidence_distribution = compute_distribution([float(v) for v in cv])

        # Degree distribution and connected components
        if column_exists(rel_df, "source_id") and column_exists(rel_df, "target_id"):
            degrees = self._compute_degrees(rel_df)
            result.degree_distribution = compute_distribution(degrees) if degrees else DistributionStats()

            # Isolated nodes
            all_nodes = set(rel_df["source_id"]).union(set(rel_df["target_id"]))
            connected_nodes = set()
            for row in rel_df.iter_rows(named=True):
                connected_nodes.add(row.get("source_id", ""))
                connected_nodes.add(row.get("target_id", ""))
            result.isolated_nodes = len(all_nodes) - len(connected_nodes)
            result.isolated_node_rate = round(safe_divide(result.isolated_nodes, max(len(all_nodes), 1)), 4)

            # Largest connected component
            lcc = self._largest_connected_component(rel_df)
            result.largest_connected_component_size = lcc
            result.largest_connected_component_pct = round(
                safe_divide(lcc, max(len(all_nodes), 1)), 4
            )

        result.health = self._compute_health(result)
        return result

    def _compute_degrees(self, rel_df: pl.DataFrame) -> list[float]:
        from collections import Counter
        degree = Counter()
        for row in rel_df.iter_rows(named=True):
            degree[row.get("source_id", "")] += 1
            degree[row.get("target_id", "")] += 1
        return [float(d) for d in degree.values()]

    def _largest_connected_component(self, rel_df: pl.DataFrame) -> int:
        adj: dict[str, set[str]] = {}
        for row in rel_df.iter_rows(named=True):
            s = row.get("source_id", "")
            t = row.get("target_id", "")
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)

        visited: set[str] = set()
        max_size = 0
        for node in adj:
            if node in visited:
                continue
            stack = [node]
            size = 0
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                size += 1
                stack.extend(adj.get(cur, set()) - visited)
            max_size = max(max_size, size)
        return max_size

    def _compute_health(self, ev: RelationshipEvaluation) -> StageHealth:
        score = 100.0
        warnings: list[str] = []

        if ev.total_relationships == 0:
            return StageHealth(score=0.0, weight=EvalConfig.weight("relationships"), warnings=["No relationships"])

        if ev.average_similarity < EvalConfig.threshold("min_similarity", 0.0):
            score -= 15
            warnings.append(f"Low average similarity: {ev.average_similarity:.4f}")

        if ev.isolated_node_rate > 0.3:
            penalty = min(20, ev.isolated_node_rate * 50)
            score -= penalty
            warnings.append(f"High isolated node rate: {ev.isolated_node_rate:.1%}")

        if ev.largest_connected_component_pct < 0.3 and ev.total_relationships > 10:
            score -= 10
            warnings.append(f"Fragmented graph: LCC covers {ev.largest_connected_component_pct:.1%}")

        if ev.confidence_distribution.mean < 0.5:
            score -= 10
            warnings.append(f"Low relationship confidence: {ev.confidence_distribution.mean:.3f}")

        score = max(0.0, min(100.0, score))
        return StageHealth(
            score=round(score, 1),
            weight=EvalConfig.weight("relationships"),
            warnings=warnings,
        )
