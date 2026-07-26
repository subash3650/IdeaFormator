"""Tests for pipeline stages (normalize, validate, enrich)."""

from __future__ import annotations

from pain_intelligence.ingestion.models import RawDocument, SourceType
from pain_intelligence.ingestion.pipeline.normalize import NormalizeStage
from pain_intelligence.ingestion.pipeline.validate import ValidateStage
from pain_intelligence.ingestion.pipeline.enrich import EnrichStage


class TestNormalizeStage:
    def test_normalize_valid_records(self):
        stage = NormalizeStage()
        records = [
            {
                "document_id": "abc123",
                "source": SourceType.GITHUB,
                "source_type": "issue",
                "external_id": "1",
                "title": "Test Issue",
                "content": "Body text",
            },
            {
                "document_id": "def456",
                "source": SourceType.GITHUB,
                "source_type": "issue",
                "external_id": "2",
                "title": "Another Issue",
                "content": "More text",
            },
        ]
        docs = stage.run(records)
        assert len(docs) == 2
        assert isinstance(docs[0], RawDocument)

    def test_normalize_invalid_record_skipped(self):
        stage = NormalizeStage()
        records = [
            {
                "document_id": "abc123",
                "source": SourceType.GITHUB,
                "source_type": "issue",
                "external_id": "1",
            },
            {
                # Missing required fields
                "title": "No ID",
            },
        ]
        docs = stage.run(records)
        assert len(docs) == 1
        assert docs[0].document_id == "abc123"

    def test_normalize_empty_batch(self):
        stage = NormalizeStage()
        docs = stage.run([])
        assert docs == []


class TestValidateStage:
    def _make_doc(self, doc_id: str = "abc123", title: str = "Test", content: str = "Body") -> RawDocument:
        return RawDocument(
            document_id=doc_id,
            source=SourceType.GITHUB,
            source_type="issue",
            external_id="1",
            title=title,
            content=content,
        )

    def test_validate_valid(self):
        stage = ValidateStage()
        doc = self._make_doc()
        valid, invalid = stage.run([doc])
        assert len(valid) == 1
        assert len(invalid) == 0

    def test_validate_duplicate_ids(self):
        stage = ValidateStage()
        doc1 = self._make_doc(doc_id="same_id")
        doc2 = self._make_doc(doc_id="same_id")
        valid, invalid = stage.run([doc1, doc2])
        assert len(valid) == 1
        assert len(invalid) == 1

    def test_validate_empty_content(self):
        stage = ValidateStage()
        doc = RawDocument(
            document_id="abc",
            source=SourceType.GITHUB,
            source_type="issue",
            external_id="1",
            title=None,
            content=None,
        )
        valid, invalid = stage.run([doc])
        assert len(invalid) == 1

    def test_validate_empty_batch(self):
        stage = ValidateStage()
        valid, invalid = stage.run([])
        assert valid == []
        assert invalid == []


class TestEnrichStage:
    def test_enrich_adds_content_length(self):
        stage = EnrichStage()
        doc = RawDocument(
            document_id="abc",
            source=SourceType.GITHUB,
            source_type="issue",
            external_id="1",
            title="Test Title",
            content="This is the body content",
        )
        enriched = stage.run([doc])
        assert len(enriched) == 1
        assert enriched[0].metadata.get("content_length") == 24
        assert enriched[0].metadata.get("title_length") == 10

    def test_enrich_preserves_existing_metadata(self):
        stage = EnrichStage()
        doc = RawDocument(
            document_id="abc",
            source=SourceType.GITHUB,
            source_type="issue",
            external_id="1",
            content="Hello",
            metadata={"custom": "value"},
        )
        enriched = stage.run([doc])
        assert enriched[0].metadata["custom"] == "value"
        assert "content_length" in enriched[0].metadata

    def test_enrich_empty_batch(self):
        stage = EnrichStage()
        enriched = stage.run([])
        assert enriched == []
