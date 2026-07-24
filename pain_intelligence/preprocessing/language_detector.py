"""Language detector.

Uses langdetect to identify document language.
"""

from __future__ import annotations

from typing import Sequence

from pain_intelligence.preprocessing.base import TextCleanerProtocol
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # Deterministic results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class LanguageDetector(TextCleanerProtocol):
    """Detect language of text and optionally filter by supported languages.

    Args:
        supported_languages: List of allowed language codes (e.g., ["en"]).
            If empty, all languages are accepted.
        min_text_length: Minimum text length for reliable detection.
    """

    def __init__(
        self,
        supported_languages: Sequence[str] | None = None,
        min_text_length: int = 20,
    ) -> None:
        self._supported = list(supported_languages or [])
        self._min_length = min_text_length

    @property
    def name(self) -> str:
        return "language_detector"

    def clean(self, text: str) -> str:
        """Return text unchanged; detection happens via detect_language method."""
        return text

    def detect_language(self, text: str) -> str | None:
        """Detect the language of the given text.

        Args:
            text: Input text.

        Returns:
            ISO 639-1 language code, or None if detection fails.
        """
        if not text or len(text) < self._min_length:
            return None

        if not LANGDETECT_AVAILABLE:
            return None

        try:
            lang = detect(text)
            return lang
        except Exception:
            return None

    def is_supported(self, language: str | None) -> bool:
        """Check if a language is in the supported list.

        If no supported languages are configured, all are accepted.
        """
        if not self._supported:
            return True
        if language is None:
            return False
        return language in self._supported
