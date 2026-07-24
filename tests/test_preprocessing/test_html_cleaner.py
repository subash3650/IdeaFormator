"""Tests for the HTML cleaner."""

from __future__ import annotations

import pytest

from pain_intelligence.preprocessing.html_cleaner import HtmlCleaner


class TestHtmlCleaner:
    """Tests for HtmlCleaner."""

    @pytest.fixture
    def cleaner(self):
        return HtmlCleaner()

    def test_name(self, cleaner):
        assert cleaner.name == "html_cleaner"

    def test_strip_simple_tag(self, cleaner):
        assert cleaner.clean("<p>Hello</p>") == "Hello"

    def test_strip_nested_tags(self, cleaner):
        assert cleaner.clean("<div><b>Bold</b></div>") == "Bold"

    def test_decode_html_entities(self, cleaner):
        result = cleaner.clean("Price: &lt; $100 &amp; tax")
        assert "< $100 & tax" in result

    def test_strip_img_tag(self, cleaner):
        result = cleaner.clean('Image: <img src="url"/> here')
        assert "<img" not in result
        assert "here" in result

    def test_empty_text(self, cleaner):
        assert cleaner.clean("") == ""

    def test_no_html(self, cleaner):
        text = "This has no HTML tags."
        assert cleaner.clean(text) == text

    def test_mixed_content(self, cleaner):
        text = "<div>Hello</div> world &amp; more"
        result = cleaner.clean(text)
        assert "<div>" not in result
        assert "Hello" in result
        assert "&" in result
