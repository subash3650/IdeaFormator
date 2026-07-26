"""Three registries for scoring, business model, and ranking providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phase3.opportunity.providers.base import (
        BusinessModelProvider,
        RankingProvider,
        ScoringProvider,
    )

# ---------------------------------------------------------------------------
# Scoring provider registry
# ---------------------------------------------------------------------------

_scoring_providers: dict[str, type[ScoringProvider]] = {}
_scoring_priorities: dict[str, int] = {}


def register_scoring_provider(name: str, priority: int = 100):
    """Decorator that registers a ScoringProvider subclass under *name*.

    Args:
        name: Canonical provider name (snake_case).
        priority: Execution priority (higher = runs first).
    """

    def wrapper(cls: type[ScoringProvider]) -> type[ScoringProvider]:
        _scoring_providers[name] = cls
        _scoring_priorities[name] = priority
        return cls

    return wrapper


def get_scoring_provider_class(name: str) -> type[ScoringProvider]:
    """Return the registered scoring provider class for *name*."""
    if name not in _scoring_providers:
        available = ", ".join(sorted(_scoring_providers))
        msg = f"Unknown scoring provider {name!r}. Available: {available}"
        raise KeyError(msg)
    return _scoring_providers[name]


def create_scoring_provider(name: str) -> ScoringProvider:
    """Instantiate a scoring provider by name."""
    cls = get_scoring_provider_class(name)
    return cls()


def available_scoring_providers() -> list[str]:
    """Return sorted list of registered scoring provider names."""
    return sorted(_scoring_providers.keys())


def scoring_provider_priority(name: str) -> int:
    """Return the priority for a named scoring provider."""
    return _scoring_priorities.get(name, 100)


def sorted_scoring_providers() -> list[str]:
    """Return providers sorted by priority descending."""
    return sorted(_scoring_providers.keys(), key=lambda n: -_scoring_priorities.get(n, 100))


# ---------------------------------------------------------------------------
# Business model provider registry
# ---------------------------------------------------------------------------

_business_model_providers: dict[str, type[BusinessModelProvider]] = {}


def register_business_model_provider(name: str):
    """Decorator that registers a BusinessModelProvider subclass under *name*."""

    def wrapper(cls: type[BusinessModelProvider]) -> type[BusinessModelProvider]:
        _business_model_providers[name] = cls
        return cls

    return wrapper


def get_business_model_provider_class(name: str) -> type[BusinessModelProvider]:
    if name not in _business_model_providers:
        available = ", ".join(sorted(_business_model_providers))
        msg = f"Unknown business model provider {name!r}. Available: {available}"
        raise KeyError(msg)
    return _business_model_providers[name]


def create_business_model_provider(name: str) -> BusinessModelProvider:
    cls = get_business_model_provider_class(name)
    return cls()


def available_business_model_providers() -> list[str]:
    return sorted(_business_model_providers.keys())


# ---------------------------------------------------------------------------
# Ranking provider registry
# ---------------------------------------------------------------------------

_ranking_providers: dict[str, type[RankingProvider]] = {}


def register_ranking_provider(name: str):
    """Decorator that registers a RankingProvider subclass under *name*."""

    def wrapper(cls: type[RankingProvider]) -> type[RankingProvider]:
        _ranking_providers[name] = cls
        return cls

    return wrapper


def get_ranking_provider_class(name: str) -> type[RankingProvider]:
    if name not in _ranking_providers:
        available = ", ".join(sorted(_ranking_providers))
        msg = f"Unknown ranking provider {name!r}. Available: {available}"
        raise KeyError(msg)
    return _ranking_providers[name]


def create_ranking_provider(name: str) -> RankingProvider:
    cls = get_ranking_provider_class(name)
    return cls()


def available_ranking_providers() -> list[str]:
    return sorted(_ranking_providers.keys())
