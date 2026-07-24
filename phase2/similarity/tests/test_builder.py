"""Tests for RelationshipBuilder."""

from __future__ import annotations

import hashlib

from phase2.embeddings.schema import SourceType
from phase2.similarity.builder import RelationshipBuilder
from phase2.similarity.config import SimilarityEngineConfig
from phase2.similarity.providers.cosine import CosineSimilarityProvider
from phase2.similarity.schema import RelationshipType


class TestRelationshipBuilder:
    def _make_builder(self) -> RelationshipBuilder:
        cfg = SimilarityEngineConfig()
        provider = CosineSimilarityProvider()
        return RelationshipBuilder(cfg, provider)

    def test_build(self) -> None:
        builder = self._make_builder()
        rel = builder.build(
            source_type=SourceType.observation,
            source_id="src1",
            target_type=SourceType.observation,
            target_id="tgt1",
            similarity_score=0.95,
            confidence=0.88,
        )
        assert rel.source_type == SourceType.observation
        assert rel.source_id == "src1"
        assert rel.target_id == "tgt1"
        assert rel.similarity_score == 0.95
        assert rel.confidence == 0.88
        assert rel.relationship_type == RelationshipType.SIMILAR
        assert rel.metric == "cosine"
        assert rel.provider == "cosine"
        assert rel.model_fingerprint == "sentence_transformers/all-MiniLM-L6-v2@384d"
        assert rel.version == "1.0"

    def test_deterministic_id(self) -> None:
        builder = self._make_builder()
        rel1 = builder.build(
            source_type=SourceType.observation,
            source_id="src1",
            target_type=SourceType.observation,
            target_id="tgt1",
            similarity_score=0.9,
            confidence=0.8,
        )
        rel2 = builder.build(
            source_type=SourceType.observation,
            source_id="src1",
            target_type=SourceType.observation,
            target_id="tgt1",
            similarity_score=0.85,
            confidence=0.7,
        )
        # Same IDs produce same relationship_id regardless of scores
        assert rel1.relationship_id == rel2.relationship_id

    def test_undirected_id(self) -> None:
        builder = self._make_builder()
        rel1 = builder.build(
            source_type=SourceType.observation,
            source_id="aaa",
            target_type=SourceType.observation,
            target_id="zzz",
            similarity_score=0.9,
            confidence=0.8,
        )
        rel2 = builder.build(
            source_type=SourceType.observation,
            source_id="zzz",
            target_type=SourceType.observation,
            target_id="aaa",
            similarity_score=0.9,
            confidence=0.8,
        )
        assert rel1.relationship_id == rel2.relationship_id

    def test_build_with_metadata(self) -> None:
        builder = self._make_builder()
        rel = builder.build(
            source_type=SourceType.observation,
            source_id="s",
            target_type=SourceType.evidence,
            target_id="t",
            similarity_score=0.5,
            confidence=0.4,
            shared_entities=["e1", "e2"],
            shared_categories=["cat1"],
            support_count=5,
            metadata={"key": "value"},
        )
        assert rel.shared_entities == ["e1", "e2"]
        assert rel.shared_categories == ["cat1"]
        assert rel.support_count == 5
        assert rel.metadata == {"key": "value"}
