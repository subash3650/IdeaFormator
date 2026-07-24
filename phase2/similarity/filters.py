"""FilterPipeline and individual filters for relationship post-processing."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phase2.embeddings.schema import SourceType
from phase2.similarity.config import SimilarityEngineConfig
from phase2.similarity.schema import SemanticRelationship


class Filter(ABC):
    """Abstract base class for relationship filters."""

    @abstractmethod
    def apply(self, relationships: list[SemanticRelationship]) -> list[SemanticRelationship]:
        """Filter the given relationships."""


class SelfSimilarityFilter(Filter):
    """Remove self-similar relationships (source_id == target_id)."""

    def apply(self, relationships: list[SemanticRelationship]) -> list[SemanticRelationship]:
        return [r for r in relationships if r.source_id != r.target_id]


class DuplicateRelationshipFilter(Filter):
    """Remove undirected duplicates (A→B and B→A) when store_bidirectional=False."""

    def __init__(self, store_bidirectional: bool = False) -> None:
        self._store_bidirectional = store_bidirectional

    def apply(self, relationships: list[SemanticRelationship]) -> list[SemanticRelationship]:
        if self._store_bidirectional:
            return relationships
        seen: set[tuple[str, str, SourceType, SourceType]] = set()
        result: list[SemanticRelationship] = []
        for r in relationships:
            key = (min(r.source_id, r.target_id), max(r.source_id, r.target_id), r.source_type, r.target_type)
            if key not in seen:
                seen.add(key)
                result.append(r)
        return result


class ThresholdFilter(Filter):
    """Remove relationships below the similarity threshold."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def apply(self, relationships: list[SemanticRelationship]) -> list[SemanticRelationship]:
        return [r for r in relationships if r.similarity_score >= self._threshold]


class ConfidenceFilter(Filter):
    """Remove relationships below the confidence threshold."""

    def __init__(self, min_confidence: float) -> None:
        self._min_confidence = min_confidence

    def apply(self, relationships: list[SemanticRelationship]) -> list[SemanticRelationship]:
        return [r for r in relationships if r.confidence >= self._min_confidence]


class RelationshipPolicyFilter(Filter):
    """Enforce allowed_relationships cross-type matrix."""

    def __init__(self, allowed: dict[SourceType, list[SourceType]]) -> None:
        self._allowed = allowed

    def apply(self, relationships: list[SemanticRelationship]) -> list[SemanticRelationship]:
        return [
            r
            for r in relationships
            if r.target_type in self._allowed.get(r.source_type, [])
        ]


class TopKPerSourceFilter(Filter):
    """Keep only the top-k relationships per source_id."""

    def __init__(self, k: int) -> None:
        self._k = k

    def apply(self, relationships: list[SemanticRelationship]) -> list[SemanticRelationship]:
        by_source: dict[str, list[SemanticRelationship]] = {}
        for r in relationships:
            by_source.setdefault(r.source_id, []).append(r)
        result: list[SemanticRelationship] = []
        for source_id, rels in by_source.items():
            sorted_rels = sorted(rels, key=lambda x: x.similarity_score, reverse=True)
            result.extend(sorted_rels[: self._k])
        return result


class FilterPipeline:
    """Composable pipeline of relationship filters."""

    def __init__(self, filters: list[Filter | None]) -> None:
        self._filters = [f for f in filters if f is not None]

    def execute(self, relationships: list[SemanticRelationship]) -> list[SemanticRelationship]:
        """Apply all filters in sequence."""
        result = relationships
        for f in self._filters:
            result = f.apply(result)
        return result

    @property
    def filter_count(self) -> int:
        return len(self._filters)
