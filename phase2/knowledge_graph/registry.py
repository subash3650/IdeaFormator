"""Decorator-based registries for node and edge builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phase2.knowledge_graph.edge_builders.base import EdgeBuilder
    from phase2.knowledge_graph.node_builders.base import NodeBuilder

_node_builders: dict[str, type[NodeBuilder]] = {}
_edge_builders: dict[str, type[EdgeBuilder]] = {}


def register_node_builder(name: str):
    """Decorator that registers a NodeBuilder subclass under `name`."""

    def wrapper(cls: type[NodeBuilder]) -> type[NodeBuilder]:
        _node_builders[name] = cls
        return cls

    return wrapper


def register_edge_builder(name: str):
    """Decorator that registers an EdgeBuilder subclass under `name`."""

    def wrapper(cls: type[EdgeBuilder]) -> type[EdgeBuilder]:
        _edge_builders[name] = cls
        return cls

    return wrapper


def get_node_builder_class(name: str) -> type[NodeBuilder]:
    if name not in _node_builders:
        available = ", ".join(sorted(_node_builders))
        msg = f"Unknown node builder {name!r}. Available: {available}"
        raise KeyError(msg)
    return _node_builders[name]


def get_edge_builder_class(name: str) -> type[EdgeBuilder]:
    if name not in _edge_builders:
        available = ", ".join(sorted(_edge_builders))
        msg = f"Unknown edge builder {name!r}. Available: {available}"
        raise KeyError(msg)
    return _edge_builders[name]


def create_node_builder(name: str, **kwargs) -> NodeBuilder:
    cls = get_node_builder_class(name)
    return cls(**kwargs)


def create_edge_builder(name: str, **kwargs) -> EdgeBuilder:
    cls = get_edge_builder_class(name)
    return cls(**kwargs)


def available_node_builders() -> list[str]:
    return sorted(_node_builders.keys())


def available_edge_builders() -> list[str]:
    return sorted(_edge_builders.keys())
