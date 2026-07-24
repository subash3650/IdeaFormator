"""Tests for the emoji processor."""

from __future__ import annotations

import pytest

from pain_intelligence.preprocessing.emoji_processor import EmojiProcessor


class TestEmojiProcessor:
    """Tests for EmojiProcessor."""

    @pytest.fixture
    def processor(self):
        return EmojiProcessor()

    def test_name(self, processor):
        assert processor.name == "emoji_processor"

    def test_remove_smile_emoji(self, processor):
        result = processor.clean("Hello 😊 World")
        assert "😊" not in result
        assert "Hello" in result

    def test_remove_multiple_emojis(self, processor):
        result = processor.clean("🎉🎈🎊 Great party!")
        assert "🎉" not in result
        assert "Great party!" in result

    def test_no_emoji(self, processor):
        text = "Plain text without any emojis."
        assert processor.clean(text) == text

    def test_empty_text(self, processor):
        assert processor.clean("") == ""

    def test_only_emojis(self, processor):
        result = processor.clean("😊😊😊")
        assert result == "" or len(result) == 0
