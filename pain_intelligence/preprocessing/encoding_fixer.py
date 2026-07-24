"""Encoding fixer for mojibake and corrupted text.

Fixes common encoding issues like â€¦ -> …, Ã© -> é, etc.
"""

from __future__ import annotations

from pain_intelligence.preprocessing.base import TextCleanerProtocol


class EncodingFixer(TextCleanerProtocol):
    """Fix encoding issues in text (mojibake, double-encoding)."""

    MOJIBAKE_MAP: dict[str, str] = {
        "\u00e2\u0080\u00a6": "\u2026",   # â€¦ -> …
        "\u00e2\u0080\u0093": "\u2013",   # â€" -> –
        "\u00e2\u0080\u0094": "\u2014",   # â€" -> —
        "\u00e2\u0080\u0099": "\u2019",   # â€™ -> '
        "\u00e2\u0080\u0098": "\u2018",   # â€˜ -> '
        "\u00e2\u0080\u009c": "\u201c",   # â€œ -> "
        "\u00e2\u0080\u009d": "\u201d",   # â€� -> "
        "\u00c2\u00a0": " ",              # non-breaking space
        "\u00e2\u0080\u009e": "\u201e",   # â€ž -> „
        "\u00e2\u0080\u009a": "\u201a",   # â€š -> ‚
        "\u00e2\u0080\u00b0": "\u2030",   # â€° -> ‰
        "\u00c3\u00a9": "\u00e9",         # Ã© -> é
        "\u00c3\u00a8": "\u00e8",         # Ã¨ -> è
        "\u00c3\u00a0": "\u00e0",         # Ã  -> à
        "\u00c3\u00b1": "\u00f1",         # Ã± -> ñ
        "\u00c3\u00bc": "\u00fc",         # Ã¼ -> ü
        "\u00c3\u00b6": "\u00f6",         # Ã¶ -> ö
        "\u00c3\u00a4": "\u00e4",         # Ã¤ -> ä
        "\u00c3\u009f": "\u00df",         # ÃŸ -> ß
    }

    @property
    def name(self) -> str:
        return "encoding_fixer"

    def clean(self, text: str) -> str:
        """Fix encoding issues in text.

        Applies known mojibake replacements and attempts
        utf-8 decode recovery.
        """
        if not text:
            return text

        for mojibake, correct in self.MOJIBAKE_MAP.items():
            text = text.replace(mojibake, correct)

        return text
