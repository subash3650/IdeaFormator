"""RelationshipBuilder – constructs SemanticRelationship objects."""

from __future__ import annotations

import hashlib
from typing import Any

from phase2.embeddings.schema import SourceType
from phase2.similarity.config import SimilarityEngineConfig
from phase2.similarity.providers.base import SimilarityProvider
from phase2.similarity.schema import RelationshipType, SemanticRelationship


class RelationshipBuilder:
    """Constructs SemanticRelationship objects with deterministic IDs."""

    def __init__(self, config: SimilarityEngineConfig, provider: SimilarityProvider) -> None:
        self._config = config
        self._provider = provider
        self._fingerprint = config.model_fingerprint

    def build(
        self,
        source_type: SourceType,
        source_id: str,
        target_type: SourceType,
        target_id: str,
        similarity_score: float,
        confidence: float,
        shared_entities: list[str] | None = None,
        shared_categories: list[str] | None = None,
        support_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticRelationship:
        """Build a single SemanticRelationship.

        Args:
            confidence: Must be computed before calling (frozen model).
        """
        relationship_id = self._make_id(source_id, target_id)
        return SemanticRelationship(
            relationship_id=relationship_id,
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            relationship_type=RelationshipType.SIMILAR,
            similarity_score=similarity_score,
            confidence=confidence,
            metric=self._config.metric,
            provider=self._provider.name,
            model_fingerprint=self._fingerprint,
            shared_entities=shared_entities or [],
            shared_categories=shared_categories or [],
            support_count=support_count,
            metadata=metadata or {},
            version=self._config.version,
        )

    def _make_id(self, source_id: str, target_id: str) -> str:
        """Deterministic relationship ID for undirected storage."""
        a, b = sorted([source_id, target_id])
        raw = f"{a}|{b}|{self._config.metric}|{self._config.version}"
        return hashlib.sha256(raw.encode()).hexdigest()
