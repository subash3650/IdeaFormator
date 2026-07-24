"""Tests for the Unicode normalizer."""

from __future__ import annotations

import pytest

from pain_intelligence.preprocessing.unicode_normalizer import UnicodeNormalizer


class TestUnicodeNormalizer:
    """Tests for UnicodeNormalizer."""

    @pytest.fixture
    def normalizer(self):
        return UnicodeNormalizer()

    def test_name(self, normalizer):
        assert normalizer.name == "unicode_normalizer"

    def test_nfc_normalization(self, normalizer):
        # Composed vs decomposed forms
        text = "cafe\u0301"  # decomposed é
        result = normalizer.clean(text)
        assert result == "caf\u00e9"  # composed é

    def test_empty_text(self, normalizer):
        assert normalizer.clean("") == ""

    def test_ascii_unchanged(self, normalizer):
        text = "Hello World 123"
        assert normalizer.clean(text) == text

    def test_unicode_preserved(self, normalizer):
        text = "Héllo Wörld ñ"
        result = normalizer.clean(text)
        assert "Héllo" in result
