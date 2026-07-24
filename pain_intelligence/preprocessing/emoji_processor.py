"""Emoji processor.

Removes or converts emoji characters from text.
"""

from __future__ import annotations

import re
import unicodedata

from pain_intelligence.preprocessing.base import TextCleanerProtocol


class EmojiProcessor(TextCleanerProtocol):
    """Remove emoji characters from text."""

    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u200d"
        "\u2640-\u2642"
        "\ufe0f"
        "\u2600-\u2B55"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\u3030"
        "\u2934"
        "\u2935"
        "]+",
        flags=re.UNICODE,
    )

    @property
    def name(self) -> str:
        return "emoji_processor"

    def clean(self, text: str) -> str:
        """Remove emoji characters from text."""
        if not text:
            return text

        return self.EMOJI_PATTERN.sub("", text).strip()
