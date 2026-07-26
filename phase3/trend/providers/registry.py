"""Provider registry for Trend Score Providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phase3.trend.providers.base import TrendScoreProvider

# ---------------------------------------------------------------------------
# Trend score provider registry
# ---------------------------------------------------------------------------

_trend_score_providers: dict[str, type[TrendScoreProvider]] = {}
_trend_score_priorities: dict[str, int] = {}


def register_trend_score_provider(name: str, priority: int = 100):
    """Decorator that registers a TrendScoreProvider subclass under *name*.

    Args:
        name: Canonical provider name (snake_case).
        priority: Execution priority (higher = runs first).
    """

    def wrapper(cls: type[TrendScoreProvider]) -> type[TrendScoreProvider]:
        _trend_score_providers[name] = cls
        _trend_score_priorities[name] = priority
        return cls

    return wrapper


def get_trend_score_provider_class(name: str) -> type[TrendScoreProvider]:
    """Return the registered trend score provider class for *name*."""
    if name not in _trend_score_providers:
        available = ", ".join(sorted(_trend_score_providers))
        msg = f"Unknown trend score provider {name!r}. Available: {available}"
        raise KeyError(msg)
    return _trend_score_providers[name]


def create_trend_score_provider(name: str) -> TrendScoreProvider:
    """Instantiate a trend score provider by name."""
    cls = get_trend_score_provider_class(name)
    return cls()


def available_trend_score_providers() -> list[str]:
    """Return sorted list of registered trend score provider names."""
    return sorted(_trend_score_providers.keys())


def trend_score_provider_priority(name: str) -> int:
    """Return the priority for a named trend score provider."""
    return _trend_score_priorities.get(name, 100)


def sorted_trend_score_providers() -> list[str]:
    """Return providers sorted by priority descending."""
    return sorted(
        _trend_score_providers.keys(),
        key=lambda n: -_trend_score_priorities.get(n, 100),
    )
