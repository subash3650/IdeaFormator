"""Abstract base classes for Opportunity Engine providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phase3.opportunity.schema import Opportunity, OpportunityType, ScoringBreakdown


class ScoringProvider(ABC):
    """Abstract base for scoring providers.

    Each provider scores candidates independently. Providers run in priority
    order and may read/update a shared ScoringBreakdown dict.
    """

    @abstractmethod
    def score(self, candidate: dict, context: dict) -> ScoringBreakdown:
        """Score a single opportunity candidate.

        Args:
            candidate: Raw candidate dict from the extractor.
            context: Shared context containing upstream data (evidence,
                     root_causes, clusters, inferences, chains, entities).

        Returns:
            ScoringBreakdown with the provider's contributions filled in.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical provider name (snake_case)."""

    @property
    @abstractmethod
    def priority(self) -> int:
        """Execution priority (higher = runs first)."""


class BusinessModelProvider(ABC):
    """Abstract base for business model providers.

    Each provider evaluates whether an opportunity matches a specific
    business model type and produces a relevance score.
    """

    @abstractmethod
    def evaluate(self, opportunity: Opportunity, context: dict) -> tuple[OpportunityType, float]:
        """Evaluate whether the opportunity matches this business model.

        Args:
            opportunity: Scored (but not yet recommended) opportunity.
            context: Shared context with upstream data.

        Returns:
            Tuple of (OpportunityType, relevance_score) where
            relevance_score in [0, 1].
        """

    @property
    @abstractmethod
    def model_type(self) -> OpportunityType:
        """The business model type this provider evaluates."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical provider name (snake_case)."""


class RankingProvider(ABC):
    """Abstract base for ranking providers."""

    @abstractmethod
    def rank(
        self, opportunities: list[Opportunity], top_k: int, context: dict
    ) -> list[Opportunity]:
        """Rank scored opportunities.

        Args:
            opportunities: Scored opportunity list.
            top_k: Maximum number to return.
            context: Shared context.

        Returns:
            Ranked list of opportunities (top_k or fewer) with rank assigned.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical provider name (snake_case)."""

    @property
    @abstractmethod
    def strategy(self) -> str:
        """Ranking strategy identifier (matches RankingStrategy enum value)."""
