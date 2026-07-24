"""Base protocol for text preprocessing modules.

All cleaners implement TextCleanerProtocol, making them
interchangeable and independently testable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TextCleanerProtocol(Protocol):
    """Interface for all text preprocessing modules."""

    @property
    def name(self) -> str:
        """Human-readable name of this cleaner."""
        ...

    def clean(self, text: str) -> str:
        """Apply cleaning transformation to the input text.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text.
        """
        ...
