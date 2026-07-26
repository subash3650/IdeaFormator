from __future__ import annotations

from typing import Any


class EvalConfig:
    _weights: dict[str, float] = {
        "documents": 0.10,
        "observations": 0.20,
        "evidence": 0.15,
        "signals": 0.15,
        "embeddings": 0.10,
        "relationships": 0.15,
        "clusters": 0.15,
    }

    _thresholds: dict[str, Any] = {
        "duplicate_rate_max": 0.10,
        "min_entity_precision": 0.0,
        "min_entity_coverage": 0.0,
        "min_extraction_rate": 0.0,
        "min_compression_ratio": 1.0,
        "min_evidence_confidence": 0.5,
        "min_signal_document_count": 3,
        "max_zero_vector_rate": 0.05,
        "max_duplicate_vector_rate": 0.10,
        "min_similarity": 0.0,
        "max_orphan_rate": 0.50,
        "max_low_quality_cluster_rate": 0.30,
        "min_cluster_quality": 0.3,
        "max_singleton_rate": 0.50,
    }

    @classmethod
    def weight(cls, stage: str) -> float:
        return cls._weights.get(stage, 0.0)

    @classmethod
    def threshold(cls, name: str, default: Any = 0.0) -> Any:
        return cls._thresholds.get(name, default)

    @classmethod
    def weights(cls) -> dict[str, float]:
        return dict(cls._weights)

    @classmethod
    def thresholds(cls) -> dict[str, Any]:
        return dict(cls._thresholds)
