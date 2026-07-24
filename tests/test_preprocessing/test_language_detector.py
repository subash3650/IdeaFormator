"""Tests for the language detector."""

from __future__ import annotations

import pytest

from pain_intelligence.preprocessing.language_detector import LanguageDetector


class TestLanguageDetector:
    """Tests for LanguageDetector."""

    @pytest.fixture
    def detector(self):
        return LanguageDetector(supported_languages=["en"])

    @pytest.fixture
    def multi_lang_detector(self):
        return LanguageDetector(supported_languages=["en", "es", "fr"])

    def test_name(self, detector):
        assert detector.name == "language_detector"

    def test_english_detected(self, detector):
        text = "This is a long enough English text for reliable detection."
        lang = detector.detect_language(text)
        assert lang == "en"

    def test_short_text_returns_none(self, detector):
        lang = detector.detect_language("Short")
        assert lang is None

    def test_empty_text_returns_none(self, detector):
        lang = detector.detect_language("")
        assert lang is None

    def test_is_supported_english(self, detector):
        assert detector.is_supported("en") is True

    def test_is_supported_french(self, detector):
        assert detector.is_supported("fr") is False

    def test_is_supported_none(self, detector):
        assert detector.is_supported(None) is False

    def test_empty_supported_accepts_all(self):
        detector = LanguageDetector(supported_languages=[])
        assert detector.is_supported("en") is True
        assert detector.is_supported("fr") is True

    def test_clean_passthrough(self, detector):
        text = "This text should pass through unchanged."
        assert detector.clean(text) == text

    def test_multi_lang(self, multi_lang_detector):
        assert multi_lang_detector.is_supported("en") is True
        assert multi_lang_detector.is_supported("es") is True
        assert multi_lang_detector.is_supported("de") is False
