"""Tests for schema models."""

from __future__ import annotations

import hashlib

from phase2.embeddings.schema import SourceType
from phase2.similarity.schema import (
    RelationshipManifest,
    RelationshipType,
    SemanticRelationship,
)


class TestSemanticRelationship:
    def test_frozen_model(self) -> None:
        rel = SemanticRelationship(
            relationship_id="abc123",
            source_type=SourceType.observation,
            source_id="src1",
            target_type=SourceType.observation,
            target_id="tgt1",
            similarity_score=0.95,
            confidence=0.88,
            metric="cosine",
            provider="cosine",
            model_fingerprint="sentence_transformers/all-MiniLM-L6-v2@384d",
            version="1.0",
        )
        assert rel.relationship_id == "abc123"
        assert rel.similarity_score == 0.95
        # Verify frozen by attempting mutation
        try:
            rel.relationship_id = "new_id"  # type: ignore
            assert False, "Should have raised"
        except Exception:
            pass

    def test_forbids_extra_fields(self) -> None:
        try:
            SemanticRelationship(
                relationship_id="abc",
                source_type=SourceType.observation,
                source_id="s",
                target_type=SourceType.observation,
                target_id="t",
                similarity_score=0.5,
                confidence=0.5,
                metric="cosine",
                provider="cosine",
                model_fingerprint="fp",
                version="1.0",
                unknown_field="bad",  # type: ignore
            )
            assert False, "Should have raised"
        except Exception:
            pass

    def test_default_relationship_type(self) -> None:
        rel = SemanticRelationship(
            relationship_id="abc",
            source_type=SourceType.observation,
            source_id="s",
            target_type=SourceType.observation,
            target_id="t",
            similarity_score=0.5,
            confidence=0.5,
            metric="cosine",
            provider="cosine",
            model_fingerprint="fp",
            version="1.0",
        )
        assert rel.relationship_type == RelationshipType.SIMILAR

    def test_score_bounds(self) -> None:
        try:
            SemanticRelationship(
                relationship_id="abc",
                source_type=SourceType.observation,
                source_id="s",
                target_type=SourceType.observation,
                target_id="t",
                similarity_score=1.5,
                confidence=0.5,
                metric="cosine",
                provider="cosine",
                model_fingerprint="fp",
                version="1.0",
            )
            assert False, "Should have raised"
        except Exception:
            pass

    def test_serialization_roundtrip(self) -> None:
        rel = SemanticRelationship(
            relationship_id="abc",
            source_type=SourceType.observation,
            source_id="s",
            target_type=SourceType.evidence,
            target_id="t",
            similarity_score=0.9,
            confidence=0.8,
            metric="cosine",
            provider="cosine",
            model_fingerprint="fp",
            version="1.0",
            shared_entities=["e1", "e2"],
        )
        data = rel.model_dump()
        rel2 = SemanticRelationship(**data)
        assert rel == rel2


class TestRelationshipManifest:
    def test_manifest_fields(self) -> None:
        m = RelationshipManifest(
            embedding_model="all-MiniLM-L6-v2",
            embedding_fingerprint="sentence_transformers/all-MiniLM-L6-v2@384d",
            metric="cosine",
            threshold=0.82,
            record_count=100,
            generated_at="2026-01-01T00:00:00",
            elapsed_seconds=5.0,
        )
        assert m.project == "pain-intelligence-engine"
        assert m.module == "relationships"
        assert m.record_count == 100
