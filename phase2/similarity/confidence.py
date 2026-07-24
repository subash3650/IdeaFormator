"""RelationshipConfidencePolicy – computes confidence scores from multiple factors.

Uses a configurable weighted geometric mean of available signals so that
confidence provides an independent measure from raw similarity alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from phase2.similarity.config import SimilarityEngineConfig


@dataclass
class ConfidenceFactors:
    """Weight configuration for each confidence factor.
    All weights should be positive.  Factors with weight 0 are skipped.
    """

    similarity_weight: float = 1.0
    source_quality_weight: float = 1.0
    target_quality_weight: float = 1.0
    source_frequency_weight: float = 1.0
    target_frequency_weight: float = 1.0
    embedding_quality_weight: float = 0.5
    support_count_weight: float = 0.5
    metadata_completeness_weight: float = 0.3


DEFAULT_WEIGHTS = ConfidenceFactors()


class RelationshipConfidencePolicy:
    """Computes confidence from multiple weighted factors.
    Uses weighted geometric mean to produce a balanced confidence score
    that penalizes missing or low-quality signals independently of similarity.
    Falls back gracefully when a factor is unavailable (defaults to 1.0).
    """

    def __init__(
        self,
        config: SimilarityEngineConfig,
        weights: ConfidenceFactors | None = None,
    ) -> None:
        self._config = config
        self._weights = weights or DEFAULT_WEIGHTS

    def compute(
        self,
        similarity_score: float,
        source_quality: float = 1.0,
        target_quality: float = 1.0,
        source_frequency: int = 1,
        target_frequency: int = 1,
        embedding_quality: float = 1.0,
        metadata_completeness: float = 1.0,
        support_count: int | None = None,
    ) -> float:
        """Compute confidence score for a single relationship.
        All provided factors should be in [0, 1]; the method clamps them.
        Missing factors default to 1.0 (neutral) or None (skipped).
        """
        factors: list[float] = []
        total_weight = 0.0

        def _add(value: float, weight: float) -> None:
            nonlocal total_weight
            if weight > 0:
                factors.append(max(0.0, min(1.0, value)) ** weight)
                total_weight += weight

        _add(similarity_score, self._weights.similarity_weight)
        _add(source_quality, self._weights.source_quality_weight)
        _add(target_quality, self._weights.target_quality_weight)

        freq_s = float(min(1.0, np.log1p(source_frequency) / np.log(100)))
        freq_t = float(min(1.0, np.log1p(target_frequency) / np.log(100)))
        _add(freq_s, self._weights.source_frequency_weight)
        _add(freq_t, self._weights.target_frequency_weight)

        _add(embedding_quality, self._weights.embedding_quality_weight)
        _add(metadata_completeness, self._weights.metadata_completeness_weight)

        if support_count is not None and support_count > 0:
            support_norm = float(min(1.0, np.log1p(support_count) / np.log(100)))
            _add(support_norm, self._weights.support_count_weight)

        if total_weight == 0:
            return similarity_score

        product = float(np.prod(factors))
        return float(np.power(product, 1.0 / total_weight))

    def compute_batch(
        self,
        similarities: list[float],
        source_qualities: list[float] | None = None,
        target_qualities: list[float] | None = None,
        source_frequencies: list[int] | None = None,
        target_frequencies: list[int] | None = None,
        embedding_qualities: list[float] | None = None,
        metadata_completeness: list[float] | None = None,
        support_counts: list[int] | None = None,
    ) -> list[float]:
        """Vectorized confidence computation for a batch of relationships."""
        n = len(similarities)

        def _arr(values, default, dtype):
            return np.full(n, default, dtype=dtype) if values is None else np.array(values, dtype=dtype)

        sims = _arr(None, 1.0, np.float32)
        sims[:] = np.array(similarities, dtype=np.float32)
        sq = _arr(source_qualities, 1.0, np.float32)
        tq = _arr(target_qualities, 1.0, np.float32)
        sf = _arr(source_frequencies, 1, np.int32)
        tf = _arr(target_frequencies, 1, np.int32)
        eq = _arr(embedding_qualities, 1.0, np.float32)
        mc = _arr(metadata_completeness, 1.0, np.float32)
        sc = _arr(support_counts, 0, np.int32)
        sc_provided = support_counts is not None

        sims = np.clip(sims, 0.0, 1.0)
        sq = np.clip(sq, 0.0, 1.0)
        tq = np.clip(tq, 0.0, 1.0)
        eq = np.clip(eq, 0.0, 1.0)
        mc = np.clip(mc, 0.0, 1.0)

        fs = np.minimum(1.0, np.log1p(sf.astype(np.float32)) / np.log(100.0))
        ft = np.minimum(1.0, np.log1p(tf.astype(np.float32)) / np.log(100.0))

        if sc_provided:
            any_positive = (sc > 0).any()
            if any_positive:
                sn = np.where(
                    sc > 0,
                    np.minimum(1.0, np.log1p(sc.astype(np.float32)) / np.log(100.0)),
                    1.0,
                )
            # If no positive support counts, skip the factor entirely

        w = self._weights
        weighted_factors = []

        def _add(f, weight):
            if weight > 0:
                weighted_factors.append(np.power(np.clip(f, 0.0, 1.0), weight))

        _add(sims, w.similarity_weight)
        _add(sq, w.source_quality_weight)
        _add(tq, w.target_quality_weight)
        _add(fs, w.source_frequency_weight)
        _add(ft, w.target_frequency_weight)
        _add(eq, w.embedding_quality_weight)
        _add(mc, w.metadata_completeness_weight)
        if sc_provided and (sc > 0).any():
            _add(sn, w.support_count_weight)

        if not weighted_factors:
            return similarities

        weight_list = [
            w.similarity_weight, w.source_quality_weight, w.target_quality_weight,
            w.source_frequency_weight, w.target_frequency_weight,
            w.embedding_quality_weight, w.metadata_completeness_weight,
        ]
        if sc_provided and (sc > 0).any():
            weight_list.append(w.support_count_weight)
        total_weight = sum(weight_list)
        stack = np.stack(weighted_factors, axis=1)
        products = np.prod(stack, axis=1)
        confidences = np.power(products, 1.0 / total_weight)
        return confidences.tolist()