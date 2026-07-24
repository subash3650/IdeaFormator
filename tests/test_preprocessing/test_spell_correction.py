"""Tests for the spell correction interface (stub)."""

from __future__ import annotations

import pytest

from pain_intelligence.preprocessing.spell_correction import SpellCorrectionInterface


class TestSpellCorrectionInterface:
    """Tests for SpellCorrectionInterface stub."""

    @pytest.fixture
    def interface(self):
        return SpellCorrectionInterface()

    def test_name(self, interface):
        assert interface.name == "spell_correction"

    def test_passthrough(self, interface):
        text = "Ths is a tset with erors"
        assert interface.clean(text) == text

    def test_empty_text(self, interface):
        assert interface.clean("") == ""

    def test_returns_string(self, interface):
        result = interface.clean("Hello world")
        assert isinstance(result, str)
