"""Abstract base class for node builders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phase2.knowledge_graph.schema import GraphNode


class NodeBuilder(ABC):
    """Builds GraphNode instances from pipeline assets."""

    @abstractmethod
    def build_nodes(self) -> list[GraphNode]:
        """Build all nodes from the source asset."""
