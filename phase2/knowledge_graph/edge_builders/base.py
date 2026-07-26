"""Abstract base class for edge builders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phase2.knowledge_graph.schema import GraphEdge, GraphNode


class EdgeBuilder(ABC):
    """Builds GraphEdge instances from graph nodes and pipeline assets."""

    @abstractmethod
    def build_edges(self, nodes: list[GraphNode]) -> list[GraphEdge]:
        """Build all edges from the available nodes and source asset."""
