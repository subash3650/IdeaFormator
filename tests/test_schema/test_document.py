"""Tests for the Document schema."""

from __future__ import annotations

from datetime import datetime

import pytest

from pain_intelligence.schema.document import (
    Document,
    Platform,
    RemovedDocument,
    RemovalReason,
)


class TestPlatform:
    """Tests for the Platform enum."""

    def test_platform_values(self):
        assert Platform.AMAZON.value == "amazon"
        assert Platform.YELP.value == "yelp"
        assert Platform.TWITTER.value == "twitter"
        assert Platform.REDDIT.value == "reddit"
        assert Platform.UNKNOWN.value == "unknown"

    def test_platform_is_string(self):
        assert isinstance(Platform.AMAZON, str)


class TestDocument:
    """Tests for the Document model."""

    def test_minimal_document(self):
        doc = Document(
            platform=Platform.AMAZON,
            source_dataset="test.csv",
            text="Hello world",
        )
        assert doc.platform == Platform.AMAZON
        assert doc.text == "Hello world"
        assert doc.id is not None
        assert doc.clean_text is None
        assert doc.document_length == 0

    def test_full_document(self):
        doc = Document(
            id="test-123",
            platform=Platform.YELP,
            source_dataset="yelp.csv",
            title="Great restaurant",
            text="Amazing food and service",
            rating=5.0,
            author="Alice",
            country="US",
            location="New York",
            language="en",
            created_at=datetime(2024, 1, 15),
            metadata={"business_id": "biz1"},
            raw_record={"stars": 5},
            clean_text="amazing food and service",
            document_length=23,
        )
        assert doc.id == "test-123"
        assert doc.rating == 5.0
        assert doc.clean_text == "amazing food and service"

    def test_rating_validation_normalizes(self):
        doc = Document(
            platform=Platform.AMAZON,
            source_dataset="test.csv",
            text="Hello",
            rating=6.0,
        )
        assert doc.rating == 5.0

    def test_rating_validation_clamps_negative(self):
        doc = Document(
            platform=Platform.AMAZON,
            source_dataset="test.csv",
            text="Hello",
            rating=-1.0,
        )
        assert doc.rating == 0.0

    def test_rating_none_allowed(self):
        doc = Document(
            platform=Platform.REDDIT,
            source_dataset="test.csv",
            text="Hello world",
            rating=None,
        )
        assert doc.rating is None

    def test_text_validation_rejects_empty(self):
        with pytest.raises(ValueError):
            Document(
                platform=Platform.AMAZON,
                source_dataset="test.csv",
                text="",
            )

    def test_text_validation_rejects_whitespace(self):
        with pytest.raises(ValueError):
            Document(
                platform=Platform.AMAZON,
                source_dataset="test.csv",
                text="   ",
            )

    def test_effective_text_prefers_clean(self):
        doc = Document(
            platform=Platform.AMAZON,
            source_dataset="test.csv",
            text="Original text",
            clean_text="Cleaned text",
        )
        assert doc.effective_text == "Cleaned text"

    def test_effective_text_falls_back_to_raw(self):
        doc = Document(
            platform=Platform.AMAZON,
            source_dataset="test.csv",
            text="Original text",
        )
        assert doc.effective_text == "Original text"

    def test_to_flat_dict(self):
        doc = Document(
            platform=Platform.AMAZON,
            source_dataset="test.csv",
            text="Hello",
            rating=4.0,
        )
        flat = doc.to_flat_dict()
        assert isinstance(flat, dict)
        assert flat["platform"] == "amazon"
        assert flat["rating"] == 4.0

    def test_metadata_default_empty(self):
        doc = Document(
            platform=Platform.AMAZON,
            source_dataset="test.csv",
            text="Hello",
        )
        assert doc.metadata == {}
        assert doc.raw_record == {}


class TestRemovedDocument:
    """Tests for the RemovedDocument model."""

    def test_removed_document(self):
        removed = RemovedDocument(
            document_id="doc-1",
            platform=Platform.AMAZON,
            source_dataset="test.csv",
            text_preview="Hello...",
            reason=RemovalReason.TOO_SHORT,
            original_length=5,
        )
        assert removed.reason == RemovalReason.TOO_SHORT

    def test_removal_reasons(self):
        assert RemovalReason.EMPTY_TEXT.value == "empty_text"
        assert RemovalReason.TOO_SHORT.value == "too_short"
        assert RemovalReason.DUPLICATE.value == "duplicate"
