from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import numpy as np

from phase2.evaluation.config import EvalConfig
from phase2.evaluation.metrics import (
    column_exists,
    compute_distribution,
    safe_divide,
)
from phase2.evaluation.schema import EmbeddingEvaluation, StageHealth


class EmbeddingEvaluator:
    EMBEDDING_FILES = [
        "embeddings_observation.parquet",
        "embeddings_evidence.parquet",
        "embeddings_problem_signal.parquet",
    ]

    def __init__(self, knowledge_dir: str | Path = "pain_intelligence/knowledge") -> None:
        self._phase2_dir = Path(knowledge_dir) / "assets" / "phase2"

    def evaluate(self) -> EmbeddingEvaluation:
        result = EmbeddingEvaluation()
        all_dfs: list[pl.DataFrame] = []
        total = 0

        for fname in self.EMBEDDING_FILES:
            path = self._phase2_dir / fname
            if path.exists():
                df = pl.read_parquet(str(path))
                source = fname.replace("embeddings_", "").replace(".parquet", "")
                result.per_source_counts[source] = df.height
                total += df.height
                all_dfs.append(df)

        result.total_vectors = total

        if total == 0 or not all_dfs:
            result.health = StageHealth(score=0.0, weight=EvalConfig.weight("embeddings"), warnings=["No embedding files"])
            return result

        combined = pl.concat(all_dfs) if len(all_dfs) > 1 else all_dfs[0]

        if column_exists(combined, "dimension"):
            dims = combined["dimension"].drop_nulls().unique().to_list()
            result.dimension = int(dims[0]) if dims else 0

        if column_exists(combined, "provider"):
            providers = combined["provider"].drop_nulls().unique().to_list()
            result.provider = providers[0] if providers else ""

        if column_exists(combined, "model"):
            models = combined["model"].drop_nulls().unique().to_list()
            result.model = models[0] if models else ""

        # Vector analysis
        if column_exists(combined, "embedding"):
            embeddings_list = combined["embedding"].to_list()

            norms = []
            zero_count = 0
            seen_hashes: set[str] = set()
            dup_count = 0

            for vec in embeddings_list:
                if vec is None:
                    continue
                arr = np.array(vec, dtype=np.float64)
                norm = float(np.linalg.norm(arr))
                norms.append(norm)
                if norm < 1e-10:
                    zero_count += 1
                vec_hash = hashlib.md5(arr.tobytes()).hexdigest()
                if vec_hash in seen_hashes:
                    dup_count += 1
                seen_hashes.add(vec_hash)

            if norms:
                result.vector_norm_distribution = compute_distribution(norms)
            result.zero_vector_count = zero_count
            result.zero_vector_rate = round(safe_divide(zero_count, total), 4)
            result.duplicate_vector_count = dup_count
            result.duplicate_vector_rate = round(safe_divide(dup_count, total), 4)

        result.health = self._compute_health(result)
        return result

    def _compute_health(self, ev: EmbeddingEvaluation) -> StageHealth:
        score = 100.0
        warnings: list[str] = []

        if ev.total_vectors == 0:
            return StageHealth(score=0.0, weight=EvalConfig.weight("embeddings"), warnings=["No embeddings"])

        if ev.dimension == 0:
            score -= 30
            warnings.append("Missing embedding dimension")

        if ev.zero_vector_rate > EvalConfig.threshold("max_zero_vector_rate", 0.05):
            penalty = min(30, ev.zero_vector_rate * 100)
            score -= penalty
            warnings.append(f"High zero-vector rate: {ev.zero_vector_rate:.1%}")

        if ev.duplicate_vector_rate > EvalConfig.threshold("max_duplicate_vector_rate", 0.10):
            penalty = min(20, ev.duplicate_vector_rate * 100)
            score -= penalty
            warnings.append(f"High duplicate vector rate: {ev.duplicate_vector_rate:.1%}")

        if not ev.model:
            score -= 10
            warnings.append("Missing embedding model info")

        score = max(0.0, min(100.0, score))
        return StageHealth(
            score=round(score, 1),
            weight=EvalConfig.weight("embeddings"),
            warnings=warnings,
        )
