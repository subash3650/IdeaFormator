"""Tests for the duplicate detector."""

from __future__ import annotations

import pytest

from pain_intelligence.preprocessing.duplicate_detector import DuplicateDetector


class TestDuplicateDetector:
    """Tests for DuplicateDetector."""

    @pytest.fixture
    def detector(self):
        return DuplicateDetector(db_path=":memory:")

    def test_name(self, detector):
        assert detector.name == "duplicate_detector"

    def test_first_occurrence_not_duplicate(self, detector):
        assert detector.is_duplicate("Hello world", "doc1") is False

    def test_second_occurrence_is_duplicate(self, detector):
        detector.is_duplicate("Hello world", "doc1")
        assert detector.is_duplicate("Hello world", "doc2") is True

    def test_different_text_not_duplicate(self, detector):
        detector.is_duplicate("Hello world", "doc1")
        assert detector.is_duplicate("Different text", "doc2") is False

    def test_case_insensitive(self, detector):
        detector.is_duplicate("Hello World", "doc1")
        assert detector.is_duplicate("hello world", "doc2") is True

    def test_whitespace_insensitive(self, detector):
        detector.is_duplicate("Hello  world", "doc1")
        assert detector.is_duplicate("Hello world", "doc2") is True

    def test_compute_hash_deterministic(self):
        h1 = DuplicateDetector.compute_hash("test")
        h2 = DuplicateDetector.compute_hash("test")
        assert h1 == h2

    def test_compute_hash_different(self):
        h1 = DuplicateDetector.compute_hash("hello")
        h2 = DuplicateDetector.compute_hash("world")
        assert h1 != h2

    def test_total_seen(self, detector):
        detector.is_duplicate("text1", "d1")
        detector.is_duplicate("text2", "d2")
        detector.is_duplicate("text1", "d3")
        # Only 2 unique hashes stored; third call was a duplicate hit
        assert detector.get_total_seen() == 2

    def test_duplicate_count(self, detector):
        detector.is_duplicate("text1", "d1")
        detector.is_duplicate("text1", "d2")
        detector.is_duplicate("text2", "d3")
        assert detector.get_duplicate_count() == 1

    def test_close(self, detector):
        detector.close()
        assert detector._con is None

    def test_clean_passthrough(self, detector):
        assert detector.clean("text") == "text"
