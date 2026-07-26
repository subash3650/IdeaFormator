"""Abstract graph interface for pluggable backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphMetadata, GraphNode, NodeType


class GraphInterface(ABC):
    """Abstract graph operations. Implement to support different backends."""

    @abstractmethod
    def add_node(self, node: GraphNode) -> None: ...

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None: ...

    @abstractmethod
    def remove_node(self, node_id: str) -> None: ...

    @abstractmethod
    def remove_edge(self, edge_id: str) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode | None: ...

    @abstractmethod
    def get_edge(self, edge_id: str) -> GraphEdge | None: ...

    @abstractmethod
    def has_node(self, node_id: str) -> bool: ...

    @abstractmethod
    def has_edge(self, edge_id: str) -> bool: ...

    @abstractmethod
    def node_count(self) -> int: ...

    @abstractmethod
    def edge_count(self) -> int: ...

    @abstractmethod
    def nodes(self) -> list[GraphNode]: ...

    @abstractmethod
    def edges(self) -> list[GraphEdge]: ...

    @abstractmethod
    def nodes_by_type(self, node_type: NodeType) -> list[GraphNode]: ...

    @abstractmethod
    def edges_by_type(self, edge_type: EdgeType) -> list[GraphEdge]: ...

    @abstractmethod
    def neighbors(self, node_id: str, edge_type: EdgeType | None = None, direction: str = "out") -> list[str]: ...

    @abstractmethod
    def predecessors(self, node_id: str, edge_type: EdgeType | None = None) -> list[str]: ...

    @abstractmethod
    def successors(self, node_id: str, edge_type: EdgeType | None = None) -> list[str]: ...

    @abstractmethod
    def out_degree(self, node_id: str) -> int: ...

    @abstractmethod
    def in_degree(self, node_id: str) -> int: ...

    @abstractmethod
    def degree(self, node_id: str) -> int: ...

    @abstractmethod
    def subgraph(self, node_ids: set[str]) -> GraphInterface: ...

    @abstractmethod
    def metadata(self, run_id: str) -> GraphMetadata: ...
