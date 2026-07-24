"""Unicode normalizer.

Normalizes Unicode text to NFC form for consistent representation.
"""

from __future__ import annotations

import unicodedata

from pain_intelligence.preprocessing.base import TextCleanerProtocol


class UnicodeNormalizer(TextCleanerProtocol):
    """Normalize Unicode text to NFC form."""

    @property
    def name(self) -> str:
        return "unicode_normalizer"

    def clean(self, text: str) -> str:
        """Normalize Unicode to NFC form."""
        if not text:
            return text

        return unicodedata.normalize("NFC", text)
