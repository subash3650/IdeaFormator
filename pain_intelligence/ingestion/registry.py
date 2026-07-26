"""Registry for data collectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pain_intelligence.ingestion.collectors.base import BaseCollector
    from pain_intelligence.ingestion.config import CollectorConfig
    from pain_intelligence.ingestion.clients.base import HttpClient

_collectors: dict[str, type[BaseCollector]] = {}


def register_collector(name: str):
    """Decorator that registers a BaseCollector subclass under `name`."""

    def wrapper(cls: type[BaseCollector]) -> type[BaseCollector]:
        _collectors[name] = cls
        return cls

    return wrapper


def get_collector_class(name: str) -> type[BaseCollector]:
    """Return the registered collector class for `name`."""
    if name not in _collectors:
        available = ", ".join(sorted(_collectors))
        msg = f"Unknown collector {name!r}. Available: {available}"
        raise KeyError(msg)
    return _collectors[name]


def create_collector(name: str, config: CollectorConfig, client: HttpClient) -> BaseCollector:
    """Instantiate a collector from configuration and HTTP client."""
    cls = get_collector_class(name)
    return cls(config=config, client=client)


def available_collectors() -> list[str]:
    """Return a sorted list of registered collector names."""
    return sorted(_collectors.keys())
