"""Tests for the whitespace cleaner."""

from __future__ import annotations

import pytest

from pain_intelligence.preprocessing.whitespace_cleaner import WhitespaceCleaner


class TestWhitespaceCleaner:
    """Tests for WhitespaceCleaner."""

    @pytest.fixture
    def cleaner(self):
        return WhitespaceCleaner()

    def test_name(self, cleaner):
        assert cleaner.name == "whitespace_cleaner"

    def test_collapse_spaces(self, cleaner):
        assert cleaner.clean("Hello   world") == "Hello world"

    def test_collapse_tabs(self, cleaner):
        assert cleaner.clean("Hello\t\tworld") == "Hello world"

    def test_collapse_newlines(self, cleaner):
        text = "Hello\n\n\n\n\nWorld"
        result = cleaner.clean(text)
        assert result.count("\n") <= 2

    def test_strip_leading_trailing(self, cleaner):
        assert cleaner.clean("  Hello  ") == "Hello"

    def test_strip_trailing_spaces(self, cleaner):
        assert cleaner.clean("Hello world  ") == "Hello world"

    def test_empty_text(self, cleaner):
        assert cleaner.clean("") == ""

    def test_preserve_single_newline(self, cleaner):
        result = cleaner.clean("Hello\nWorld")
        assert "Hello\nWorld" == result
