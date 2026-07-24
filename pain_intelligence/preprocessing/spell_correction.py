"""Spell correction interface (stub).

Provides the interface for future spell correction implementation.
Currently a no-op pass-through.
"""

from __future__ import annotations

from pain_intelligence.preprocessing.base import TextCleanerProtocol


class SpellCorrectionInterface(TextCleanerProtocol):
    """Spell correction stub.

    Implements TextCleanerProtocol as a no-op.
    Replace this with a real implementation (e.g., pyspellchecker, SymSpell)
    when needed.
    """

    @property
    def name(self) -> str:
        return "spell_correction"

    def clean(self, text: str) -> str:
        """No-op: return text unchanged.

        Future implementations should correct misspellings here.
        """
        return text
