from __future__ import annotations

from typing import Any

from phase3.presentation.templates.base import BaseTemplate

_templates: dict[str, type[BaseTemplate]] = {}


def register_template(name: str):
    def decorator(cls: type[BaseTemplate]) -> type[BaseTemplate]:
        _templates[name] = cls
        return cls
    return decorator


def get_template_class(name: str) -> type[BaseTemplate]:
    if name not in _templates:
        available = ", ".join(sorted(_templates))
        msg = f"Unknown template '{name}'. Available: {available}"
        raise KeyError(msg)
    return _templates[name]


def create_template(name: str, **kwargs: Any) -> BaseTemplate:
    cls = get_template_class(name)
    return cls(**kwargs)


def available_templates() -> list[str]:
    return sorted(_templates)
