"""Abstract base class for Trend Score Providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phase3.trend.schema import TrendScoringBreakdown


class TrendScoreProvider(ABC):
    """Abstract base for trend scoring providers.

    Each provider computes a specific dimension of the trend score.
    Providers run in priority order (highest first) and may overlap.
    """

    @abstractmethod
    def score(self, candidate: dict, context: dict) -> TrendScoringBreakdown:
        """Score a single trend candidate.

        Args:
            candidate: Raw candidate dict from the trend extractor.
            context: Shared context with upstream data.

        Returns:
            TrendScoringBreakdown with the provider's contributions.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical provider name (snake_case)."""

    @property
    @abstractmethod
    def priority(self) -> int:
        """Execution priority (higher = runs first)."""
