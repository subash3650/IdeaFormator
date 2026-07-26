"""Registry for cluster providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phase2.clustering.config import ClusteringConfig
    from phase2.clustering.providers.base import ClusterProvider

_providers: dict[str, type[ClusterProvider]] = {}


def register_provider(name: str):
    """Decorator that registers a ClusterProvider subclass under `name`."""

    def wrapper(cls: type[ClusterProvider]) -> type[ClusterProvider]:
        _providers[name] = cls
        return cls

    return wrapper


def get_provider_class(name: str) -> type[ClusterProvider]:
    """Return the registered provider class for `name`."""
    if name not in _providers:
        available = ", ".join(sorted(_providers))
        msg = f"Unknown provider {name!r}. Available: {available}"
        raise KeyError(msg)
    return _providers[name]


def create_provider(config: ClusteringConfig) -> ClusterProvider:
    """Instantiate a provider from a clustering config."""
    cls = get_provider_class(config.provider)
    return cls()


def available_providers() -> list[str]:
    """Return a sorted list of registered provider names."""
    return sorted(_providers.keys())
