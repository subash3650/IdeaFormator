"""Tests for the encoding fixer."""

from __future__ import annotations

import pytest

from pain_intelligence.preprocessing.encoding_fixer import EncodingFixer


class TestEncodingFixer:
    """Tests for EncodingFixer."""

    @pytest.fixture
    def fixer(self):
        return EncodingFixer()

    def test_name(self, fixer):
        assert fixer.name == "encoding_fixer"

    def test_fix_mojibake_ellipsis(self, fixer):
        text = "This is bad\u00e2\u0080\u00a6"
        result = fixer.clean(text)
        assert "\u2026" in result

    def test_fix_mojibake_em_dash(self, fixer):
        text = "Hello\u00e2\u0080\u0094world"
        result = fixer.clean(text)
        assert "\u2014" in result

    def test_fix_mojibake_curly_quotes(self, fixer):
        text = "It\u00e2\u0080\u0099s a test"
        result = fixer.clean(text)
        assert "\u2019" in result

    def test_clean_normal_text(self, fixer):
        text = "This is normal text with no issues."
        assert fixer.clean(text) == text

    def test_empty_text(self, fixer):
        assert fixer.clean("") == ""

    def test_none_like_text(self, fixer):
        assert fixer.clean("Normal ASCII text") == "Normal ASCII text"

    def test_fix_nbsp(self, fixer):
        text = "Hello\u00c2\u00a0world"
        result = fixer.clean(text)
        assert "\u00a0" not in result
