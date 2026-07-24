"""Text normalizer.

Applies configurable normalization: lowercasing, special char removal, etc.
"""

from __future__ import annotations

import re

from pain_intelligence.preprocessing.base import TextCleanerProtocol


class TextNormalizer(TextCleanerProtocol):
    """Normalize text with configurable transformations.

    Args:
        lowercase: Whether to convert text to lowercase.
        remove_special_chars: Whether to remove non-alphanumeric characters.
    """

    SPECIAL_CHARS = re.compile(r"[^a-zA-Z0-9\s.,?'-]")

    def __init__(
        self,
        lowercase: bool = False,
        remove_special_chars: bool = False,
    ) -> None:
        self._lowercase = lowercase
        self._remove_special = remove_special_chars

    @property
    def name(self) -> str:
        return "text_normalizer"

    def clean(self, text: str) -> str:
        """Apply normalization transforms."""
        if not text:
            return text

        if self._lowercase:
            text = text.lower()
        if self._remove_special:
            text = self.SPECIAL_CHARS.sub(" ", text)
            text = re.sub(r"\s+", " ", text).strip()

        return text
