"""Tests for the text normalizer."""

from __future__ import annotations

import pytest

from pain_intelligence.preprocessing.text_normalizer import TextNormalizer


class TestTextNormalizer:
    """Tests for TextNormalizer."""

    def test_name_default(self):
        normalizer = TextNormalizer()
        assert normalizer.name == "text_normalizer"

    def test_no_transform(self):
        normalizer = TextNormalizer()
        text = "Hello World!"
        assert normalizer.clean(text) == text

    def test_lowercase(self):
        normalizer = TextNormalizer(lowercase=True)
        assert normalizer.clean("Hello WORLD") == "hello world"

    def test_remove_special_chars(self):
        normalizer = TextNormalizer(remove_special_chars=True)
        result = normalizer.clean("Hello @World# $100!")
        assert "@" not in result
        assert "#" not in result

    def test_lowercase_and_special(self):
        normalizer = TextNormalizer(lowercase=True, remove_special_chars=True)
        result = normalizer.clean("Test@Data 123!")
        assert result == "test data 123"

    def test_empty_text(self):
        normalizer = TextNormalizer()
        assert normalizer.clean("") == ""

    def test_preserve_punctuation(self):
        normalizer = TextNormalizer()
        text = "Hello, world. How are you?"
        assert normalizer.clean(text) == text
