"""Forward and reverse adjacency indexes for O(1) neighbor lookups."""

from __future__ import annotations

from phase2.knowledge_graph.schema import EdgeType, GraphEdge


class AdjacencyIndex:
    """Forward adjacency index: node_id -> {edge_id: target_node_id}.

    Maintains both a flat index and a typed index for edge-type-filtered lookups.
    """

    def __init__(self) -> None:
        self._forward: dict[str, dict[str, str]] = {}
        self._forward_typed: dict[str, dict[str, dict[str, str]]] = {}

    def add(self, edge: GraphEdge) -> None:
        """Register a directed edge source -> target."""
        self._forward.setdefault(edge.source_node_id, {})[edge.edge_id] = edge.target_node_id
        self._forward_typed.setdefault(edge.source_node_id, {}).setdefault(edge.edge_type.value, {})[
            edge.edge_id
        ] = edge.target_node_id

    def remove(self, edge_id: str, edge: GraphEdge | None = None) -> None:
        """Remove an edge from the index by edge_id."""
        if edge is not None:
            self._remove_from_node(edge.source_node_id, edge_id, edge.edge_type)
            return
        for node_id in list(self._forward.keys()):
            self._remove_from_node(node_id, edge_id, None)

    def _remove_from_node(self, node_id: str, edge_id: str, edge_type: EdgeType | None) -> None:
        flat = self._forward.get(node_id)
        if flat and edge_id in flat:
            del flat[edge_id]
        if edge_type:
            typed = self._forward_typed.get(node_id, {}).get(edge_type.value, {})
            typed.pop(edge_id, None)

    def remove_node(self, node_id: str) -> None:
        """Remove all outgoing edges for a node."""
        self._forward.pop(node_id, None)
        self._forward_typed.pop(node_id, None)

    def neighbors(self, node_id: str, edge_type: EdgeType | None = None) -> list[str]:
        """Return target node IDs for outgoing edges from node_id."""
        if edge_type:
            return list(self._forward_typed.get(node_id, {}).get(edge_type.value, {}).values())
        return list(self._forward.get(node_id, {}).values())

    def out_degree(self, node_id: str) -> int:
        """Return the number of outgoing edges from node_id."""
        return len(self._forward.get(node_id, {}))

    def has_edge(self, edge_id: str) -> bool:
        """Check if an edge exists in the forward index."""
        return any(edge_id in edges for edges in self._forward.values())

    def edge_target(self, edge_id: str) -> str | None:
        """Return the target node for a given edge_id."""
        for edges in self._forward.values():
            if edge_id in edges:
                return edges[edge_id]
        return None

    def clear(self) -> None:
        self._forward.clear()
        self._forward_typed.clear()


class ReverseAdjacencyIndex:
    """Reverse adjacency index: node_id -> {edge_id: source_node_id}.

    Maintains both a flat index and a typed index.
    """

    def __init__(self) -> None:
        self._reverse: dict[str, dict[str, str]] = {}
        self._reverse_typed: dict[str, dict[str, dict[str, str]]] = {}

    def add(self, edge: GraphEdge) -> None:
        """Register a directed edge target <- source."""
        self._reverse.setdefault(edge.target_node_id, {})[edge.edge_id] = edge.source_node_id
        self._reverse_typed.setdefault(edge.target_node_id, {}).setdefault(edge.edge_type.value, {})[
            edge.edge_id
        ] = edge.source_node_id

    def remove(self, edge_id: str, edge: GraphEdge | None = None) -> None:
        """Remove an edge from the index by edge_id."""
        if edge is not None:
            self._remove_from_node(edge.target_node_id, edge_id, edge.edge_type)
            return
        for node_id in list(self._reverse.keys()):
            self._remove_from_node(node_id, edge_id, None)

    def _remove_from_node(self, node_id: str, edge_id: str, edge_type: EdgeType | None) -> None:
        flat = self._reverse.get(node_id)
        if flat and edge_id in flat:
            del flat[edge_id]
        if edge_type:
            typed = self._reverse_typed.get(node_id, {}).get(edge_type.value, {})
            typed.pop(edge_id, None)

    def remove_node(self, node_id: str) -> None:
        """Remove all incoming edges for a node."""
        self._reverse.pop(node_id, None)
        self._reverse_typed.pop(node_id, None)

    def predecessors(self, node_id: str, edge_type: EdgeType | None = None) -> list[str]:
        """Return source node IDs for incoming edges to node_id."""
        if edge_type:
            return list(self._reverse_typed.get(node_id, {}).get(edge_type.value, {}).values())
        return list(self._reverse.get(node_id, {}).values())

    def in_degree(self, node_id: str) -> int:
        """Return the number of incoming edges to node_id."""
        return len(self._reverse.get(node_id, {}))

    def has_edge(self, edge_id: str) -> bool:
        """Check if an edge exists in the reverse index."""
        return any(edge_id in edges for edges in self._reverse.values())

    def edge_source(self, edge_id: str) -> str | None:
        """Return the source node for a given edge_id."""
        for edges in self._reverse.values():
            if edge_id in edges:
                return edges[edge_id]
        return None

    def clear(self) -> None:
        self._reverse.clear()
        self._reverse_typed.clear()
