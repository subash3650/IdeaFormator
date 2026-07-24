"""Tests for the URL cleaner."""

from __future__ import annotations

import pytest

from pain_intelligence.preprocessing.url_cleaner import UrlCleaner


class TestUrlCleaner:
    """Tests for UrlCleaner."""

    @pytest.fixture
    def cleaner(self):
        return UrlCleaner()

    def test_name(self, cleaner):
        assert cleaner.name == "url_cleaner"

    def test_remove_http_url(self, cleaner):
        result = cleaner.clean("Visit https://example.com for more")
        assert "https://example.com" not in result
        assert "[URL]" in result

    def test_remove_www_url(self, cleaner):
        result = cleaner.clean("Check www.example.com today")
        assert "www.example.com" not in result

    def test_no_url(self, cleaner):
        text = "This has no URLs at all."
        assert cleaner.clean(text) == text

    def test_empty_text(self, cleaner):
        assert cleaner.clean("") == ""

    def test_multiple_urls(self, cleaner):
        result = cleaner.clean("Go to https://a.com and https://b.com")
        assert "https://a.com" not in result
        assert "https://b.com" not in result
