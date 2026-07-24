"""HTML tag cleaner.

Strips HTML tags and decodes HTML entities from text.
"""

from __future__ import annotations

import html
import re

from pain_intelligence.preprocessing.base import TextCleanerProtocol


class HtmlCleaner(TextCleanerProtocol):
    """Remove HTML tags and decode HTML entities."""

    TAG_PATTERN = re.compile(r"<[^>]+>")
    ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;|&#\d+;")

    @property
    def name(self) -> str:
        return "html_cleaner"

    def clean(self, text: str) -> str:
        """Remove HTML tags and decode entities."""
        if not text:
            return text

        text = self.TAG_PATTERN.sub(" ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
