"""Tests for ingestion data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pain_intelligence.ingestion.models import (
    CollectionResult,
    IngestionManifest,
    RawDocument,
    SourceType,
    SyncState,
)


class TestSourceType:
    def test_source_type_values(self):
        assert SourceType.GITHUB == "github"
        assert SourceType.HACKERNEWS == "hackernews"
        assert SourceType.PRODUCTHUNT == "producthunt"
        assert SourceType.YOUTUBE == "youtube"
        assert SourceType.PLAYSTORE == "playstore"

    def test_source_type_is_str(self):
        assert isinstance(SourceType.GITHUB, str)
        assert SourceType.GITHUB == "github"


class TestRawDocument:
    def test_minimal_document(self):
        doc = RawDocument(
            document_id="abc123",
            source=SourceType.GITHUB,
            source_type="issue",
            external_id="999",
        )
        assert doc.document_id == "abc123"
        assert doc.source == SourceType.GITHUB
        assert doc.schema_version == "1.0.0"
        assert doc.tags == []
        assert doc.metadata == {}

    def test_full_document(self):
        now = datetime.now(timezone.utc)
        doc = RawDocument(
            document_id="abc123",
            source=SourceType.HACKERNEWS,
            source_type="story",
            external_id="40001",
            title="Test Title",
            content="Test content here",
            author="user1",
            created_at=now,
            language="en",
            url="https://example.com",
            tags=["tag1", "tag2"],
            categories=["cat1"],
            metadata={"score": 100},
            checksum="sha256hash",
        )
        assert doc.title == "Test Title"
        assert doc.author == "user1"
        assert len(doc.tags) == 2
        assert doc.metadata["score"] == 100

    def test_frozen_enforcement(self):
        doc = RawDocument(
            document_id="abc123",
            source=SourceType.GITHUB,
            source_type="issue",
            external_id="999",
        )
        with pytest.raises(ValidationError):
            doc.document_id = "changed"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            RawDocument(
                document_id="abc123",
                source=SourceType.GITHUB,
                source_type="issue",
                external_id="999",
                unknown_field="should fail",
            )

    def test_to_flat_dict(self):
        doc = RawDocument(
            document_id="abc123",
            source=SourceType.GITHUB,
            source_type="issue",
            external_id="999",
            title="Test",
            tags=["bug", "critical"],
            categories=["repo:test"],
        )
        flat = doc.to_flat_dict()
        assert flat["source"] == "github"
        assert flat["tags"] == "bug,critical"
        assert flat["categories"] == "repo:test"
        assert isinstance(flat["metadata"], str)

    def test_document_id_deterministic(self):
        """Same source + external_id should produce the same document_id."""
        from pain_intelligence.ingestion.utils import compute_document_id

        id1 = compute_document_id("github", "12345")
        id2 = compute_document_id("github", "12345")
        assert id1 == id2

        id3 = compute_document_id("hackernews", "12345")
        assert id1 != id3


class TestCollectionResult:
    def test_defaults(self):
        result = CollectionResult(source=SourceType.GITHUB)
        assert result.documents_collected == 0
        assert result.errors == []

    def test_frozen(self):
        result = CollectionResult(source=SourceType.GITHUB)
        with pytest.raises(ValidationError):
            result.documents_collected = 5


class TestSyncState:
    def test_defaults(self):
        state = SyncState(source=SourceType.GITHUB)
        assert state.last_sync is None
        assert state.failure_count == 0
        assert state.total_collected == 0

    def test_frozen(self):
        state = SyncState(source=SourceType.GITHUB)
        with pytest.raises(ValidationError):
            state.failure_count = 1


class TestIngestionManifest:
    def test_manifest(self):
        manifest = IngestionManifest(source=SourceType.GITHUB, document_count=100)
        assert manifest.schema_version == "1.0.0"
        assert manifest.document_count == 100
        assert manifest.generated_at is not None
