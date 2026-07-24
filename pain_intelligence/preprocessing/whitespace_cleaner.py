"""Whitespace cleaner.

Normalizes whitespace, removes duplicate spaces, and strips leading/trailing.
"""

from __future__ import annotations

import re

from pain_intelligence.preprocessing.base import TextCleanerProtocol


class WhitespaceCleaner(TextCleanerProtocol):
    """Normalize whitespace in text."""

    MULTI_SPACE = re.compile(r"[ \t]+")
    MULTI_NEWLINE = re.compile(r"\n{3,}")
    TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)

    @property
    def name(self) -> str:
        return "whitespace_cleaner"

    def clean(self, text: str) -> str:
        """Normalize whitespace: collapse spaces, limit newlines, strip."""
        if not text:
            return text

        text = self.MULTI_SPACE.sub(" ", text)
        text = self.TRAILING_SPACE.sub("", text)
        text = self.MULTI_NEWLINE.sub("\n\n", text)
        text = text.strip()
        return text
