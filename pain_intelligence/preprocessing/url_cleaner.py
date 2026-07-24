"""URL cleaner.

Removes or replaces URLs in text.
"""

from __future__ import annotations

import re

from pain_intelligence.preprocessing.base import TextCleanerProtocol


class UrlCleaner(TextCleanerProtocol):
    """Remove URLs from text."""

    URL_PATTERN = re.compile(
        r"https?://[^\s<>\"')\]]+"
        r"|www\.[^\s<>\"')\]]+"
        r"|[a-zA-Z0-9._-]+\.(?:com|org|net|edu|gov|io|co)\S*",
        re.IGNORECASE,
    )

    @property
    def name(self) -> str:
        return "url_cleaner"

    def clean(self, text: str) -> str:
        """Remove URLs from text."""
        if not text:
            return text

        return self.URL_PATTERN.sub(" [URL] ", text).strip()
