"""Provider registry with a lightweight registrar decorator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phase2.similarity.providers.base import SimilarityProvider
    from phase2.similarity.config import SimilarityEngineConfig

_providers: dict[str, type[SimilarityProvider]] = {}


def register(name: str):
    """Decorator that registers a SimilarityProvider subclass under *name*."""

    def wrapper(cls: type[SimilarityProvider]) -> type[SimilarityProvider]:
        _providers[name] = cls
        return cls

    return wrapper


def get_provider_class(name: str) -> type[SimilarityProvider]:
    """Return the registered provider class for *name*."""
    if name not in _providers:
        available = ", ".join(sorted(_providers))
        msg = f"Unknown provider {name!r}. Available: {available}"
        raise KeyError(msg)
    return _providers[name]


def create_provider(config: SimilarityEngineConfig) -> SimilarityProvider:
    """Instantiate a provider from an engine config."""
    cls = get_provider_class(config.metric)
    return cls()


def available_providers() -> list[str]:
    """Return sorted list of registered provider names."""
    return sorted(_providers.keys())
