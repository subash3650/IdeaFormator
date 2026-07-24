"""Tests for filter pipeline and individual filters."""

from __future__ import annotations

import pytest

from phase2.embeddings.schema import SourceType
from phase2.similarity.filters import (
    ConfidenceFilter,
    DuplicateRelationshipFilter,
    FilterPipeline,
    RelationshipPolicyFilter,
    SelfSimilarityFilter,
    ThresholdFilter,
    TopKPerSourceFilter,
)
from phase2.similarity.schema import SemanticRelationship


def _make_rel(
    source_id: str = "s",
    target_id: str = "t",
    source_type: SourceType = SourceType.observation,
    target_type: SourceType = SourceType.observation,
    similarity: float = 0.9,
    confidence: float = 0.8,
) -> SemanticRelationship:
    return SemanticRelationship(
        relationship_id=f"rel_{source_id}_{target_id}",
        source_type=source_type,
        source_id=source_id,
        target_type=target_type,
        target_id=target_id,
        similarity_score=similarity,
        confidence=confidence,
        metric="cosine",
        provider="cosine",
        model_fingerprint="fp",
        version="1.0",
    )


class TestSelfSimilarityFilter:
    def test_removes_self(self) -> None:
        f = SelfSimilarityFilter()
        rels = [_make_rel(source_id="a", target_id="a"), _make_rel(source_id="a", target_id="b")]
        result = f.apply(rels)
        assert len(result) == 1
        assert result[0].target_id == "b"

    def test_keeps_all_different(self) -> None:
        f = SelfSimilarityFilter()
        rels = [_make_rel(source_id="a", target_id="b"), _make_rel(source_id="c", target_id="d")]
        assert len(f.apply(rels)) == 2


class TestDuplicateRelationshipFilter:
    def test_removes_duplicates(self) -> None:
        f = DuplicateRelationshipFilter(store_bidirectional=False)
        rels = [
            _make_rel(source_id="a", target_id="b"),
            _make_rel(source_id="b", target_id="a"),
        ]
        result = f.apply(rels)
        assert len(result) == 1

    def test_keeps_bidirectional(self) -> None:
        f = DuplicateRelationshipFilter(store_bidirectional=True)
        rels = [
            _make_rel(source_id="a", target_id="b"),
            _make_rel(source_id="b", target_id="a"),
        ]
        result = f.apply(rels)
        assert len(result) == 2


class TestThresholdFilter:
    def test_removes_below_threshold(self) -> None:
        f = ThresholdFilter(0.85)
        rels = [
            _make_rel(similarity=0.9),
            _make_rel(similarity=0.8),
            _make_rel(similarity=0.85),
        ]
        result = f.apply(rels)
        assert len(result) == 2


class TestConfidenceFilter:
    def test_removes_below_confidence(self) -> None:
        f = ConfidenceFilter(0.75)
        rels = [
            _make_rel(confidence=0.9),
            _make_rel(confidence=0.5),
        ]
        result = f.apply(rels)
        assert len(result) == 1


class TestRelationshipPolicyFilter:
    def test_filters_cross_type(self) -> None:
        allowed = {SourceType.observation: [SourceType.observation]}
        f = RelationshipPolicyFilter(allowed)
        rels = [
            _make_rel(source_type=SourceType.observation, target_type=SourceType.observation),
            _make_rel(source_type=SourceType.observation, target_type=SourceType.evidence),
        ]
        result = f.apply(rels)
        assert len(result) == 1


class TestTopKPerSourceFilter:
    def test_keeps_top_k(self) -> None:
        f = TopKPerSourceFilter(k=2)
        rels = [
            _make_rel(source_id="a", target_id="t1", similarity=0.9),
            _make_rel(source_id="a", target_id="t2", similarity=0.8),
            _make_rel(source_id="a", target_id="t3", similarity=0.7),
        ]
        result = f.apply(rels)
        assert len(result) == 2
        assert result[0].similarity_score == 0.9


class TestFilterPipeline:
    def test_composes_filters(self) -> None:
        pipeline = FilterPipeline([
            SelfSimilarityFilter(),
            ThresholdFilter(0.85),
        ])
        rels = [
            _make_rel(source_id="a", target_id="a", similarity=0.9),
            _make_rel(source_id="a", target_id="b", similarity=0.9),
            _make_rel(source_id="a", target_id="c", similarity=0.8),
        ]
        result = pipeline.execute(rels)
        assert len(result) == 1

    def test_empty_pipeline(self) -> None:
        pipeline = FilterPipeline([])
        rels = [_make_rel()]
        assert len(pipeline.execute(rels)) == 1

    def test_filters_none_values(self) -> None:
        pipeline = FilterPipeline([None, SelfSimilarityFilter(), None])  # type: ignore
        assert pipeline.filter_count == 1
