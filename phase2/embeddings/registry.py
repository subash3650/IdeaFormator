"""Provider registry with a lightweight registrar decorator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phase2.embeddings.providers.base import EmbeddingProvider
    from phase2.embeddings.config import EmbeddingEngineConfig

_providers: dict[str, type[EmbeddingProvider]] = {}


def register(name: str):
    """Decorator that registers an EmbeddingProvider subclass under *name*."""
    def wrapper(cls: type[EmbeddingProvider]) -> type[EmbeddingProvider]:
        _providers[name] = cls
        return cls
    return wrapper


def get_provider_class(name: str) -> type[EmbeddingProvider]:
    """Return the registered provider class for *name*."""
    if name not in _providers:
        available = ", ".join(sorted(_providers))
        msg = f"Unknown provider {name!r}. Available: {available}"
        raise KeyError(msg)
    return _providers[name]


def create_provider(config: EmbeddingEngineConfig) -> EmbeddingProvider:
    """Instantiate a provider from an engine config."""
    cls = get_provider_class(config.provider)
    return cls(config)