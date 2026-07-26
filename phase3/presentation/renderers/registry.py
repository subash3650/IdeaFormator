from __future__ import annotations

from phase3.presentation.renderers.base import Renderer

_renderers: dict[str, type[Renderer]] = {}


def register_renderer(name: str):
    def decorator(cls: type[Renderer]) -> type[Renderer]:
        _renderers[name] = cls
        return cls
    return decorator


def get_renderer_class(name: str) -> type[Renderer]:
    if name not in _renderers:
        available = ", ".join(sorted(_renderers))
        msg = f"Unknown renderer '{name}'. Available: {available}"
        raise KeyError(msg)
    return _renderers[name]


def create_renderer(name: str) -> Renderer:
    cls = get_renderer_class(name)
    return cls()


def available_renderers() -> list[str]:
    return sorted(_renderers)
