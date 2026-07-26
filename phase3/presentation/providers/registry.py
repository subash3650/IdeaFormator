from __future__ import annotations

from typing import Any

from phase3.presentation.providers.base import ChartProvider

_providers: dict[str, type[ChartProvider]] = {}
_priorities: dict[str, int] = {}


def register_chart_provider(name: str, priority: int = 100):
    def decorator(cls: type[ChartProvider]) -> type[ChartProvider]:
        _providers[name] = cls
        _priorities[name] = priority
        return cls
    return decorator


def get_chart_provider_class(name: str) -> type[ChartProvider]:
    if name not in _providers:
        available = ", ".join(sorted(_providers))
        msg = f"Unknown chart provider '{name}'. Available: {available}"
        raise KeyError(msg)
    return _providers[name]


def create_chart_provider(name: str, **kwargs: Any) -> ChartProvider:
    cls = get_chart_provider_class(name)
    return cls(**kwargs)


def available_chart_providers() -> list[str]:
    return sorted(_providers)


def sorted_chart_providers() -> list[str]:
    return sorted(_providers, key=lambda n: -_priorities.get(n, 100))


def chart_provider_priority(name: str) -> int:
    return _priorities.get(name, 100)
