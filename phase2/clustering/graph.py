"""Relationship Graph — lightweight graph abstraction for semantic clustering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class RelationshipEdge(BaseModel):
    """Immutable edge representing a semantic relationship between two concepts."""

    source_id: str = Field(description="Source concept ID")
    target_id: str = Field(description="Target concept ID")
    similarity: float = Field(ge=0.0, le=1.0, description="Similarity score")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    relationship_type: str = Field(default="similar", description="Relationship classification")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata")

    model_config = {"frozen": True, "extra": "forbid"}

    @property
    def sorted_pair(self) -> tuple[str, str]:
        """Return sorted (source, target) tuple for undirected operations."""
        return tuple(sorted([self.source_id, self.target_id]))


class RelationshipGraph:
    """Lightweight graph for semantic relationships.

    Nodes = semantic concepts (member IDs)
    Edges = semantic relationships with weights (similarity scores)

    Complexity:
        - Construction: O(E)
        - Memory: O(V + E) adjacency list
    """

    def __init__(self) -> None:
        self._nodes: set[str] = set()
        self._adjacency: dict[str, dict[str, float]] = defaultdict(dict)  # node -> {neighbor: weight}
        self._edges: dict[tuple[str, str], RelationshipEdge] = {}  # sorted pair -> edge
        self._degrees: dict[str, int] = defaultdict(int)

    @classmethod
    def from_edges(cls, edges: list[RelationshipEdge]) -> RelationshipGraph:
        """Build graph from a list of RelationshipEdge objects."""
        graph = cls()
        for edge in edges:
            graph.add_edge(edge)
        return graph

    def add_edge(self, edge: RelationshipEdge) -> None:
        """Add a relationship edge to the graph."""
        u, v = edge.source_id, edge.target_id
        weight = edge.similarity
        pair = tuple(sorted([u, v]))

        self._nodes.add(u)
        self._nodes.add(v)
        self._adjacency[u][v] = weight
        self._adjacency[v][u] = weight
        self._edges[pair] = edge
        self._degrees[u] += 1
        self._degrees[v] += 1

    def add_edges(self, edges: list[RelationshipEdge]) -> None:
        """Add multiple edges efficiently."""
        for edge in edges:
            self.add_edge(edge)

    def nodes(self) -> set[str]:
        """Return all nodes in the graph."""
        return self._nodes.copy()

    def edges(self) -> list[RelationshipEdge]:
        """Return all edges in the graph."""
        return list(self._edges.values())

    def neighbors(self, node: str) -> dict[str, float]:
        """Return neighbors of a node with edge weights.

        Returns:
            dict mapping neighbor_id -> similarity weight
        """
        return self._adjacency.get(node, {}).copy()

    def degree(self, node: str) -> int:
        """Return the degree (number of neighbors) of a node."""
        return self._degrees.get(node, 0)

    def weighted_degree(self, node: str) -> float:
        """Return the sum of edge weights for a node."""
        return sum(self._adjacency.get(node, {}).values())

    def edge_weight(self, u: str, v: str) -> float | None:
        """Return the weight of edge (u, v) if it exists."""
        pair = tuple(sorted([u, v]))
        edge = self._edges.get(pair)
        return edge.similarity if edge else None

    def has_edge(self, u: str, v: str) -> bool:
        """Check if an edge exists between two nodes."""
        pair = tuple(sorted([u, v]))
        return pair in self._edges

    def get_edge(self, u: str, v: str) -> RelationshipEdge | None:
        """Return the RelationshipEdge between two nodes."""
        pair = tuple(sorted([u, v]))
        return self._edges.get(pair)

    def subgraph(self, node_subset: set[str]) -> RelationshipGraph:
        """Return an induced subgraph containing only the specified nodes."""
        sub = RelationshipGraph()
        for node in node_subset:
            if node in self._nodes:
                sub._nodes.add(node)
                sub._degrees[node] = 0
        for edge in self._edges.values():
            if edge.source_id in node_subset and edge.target_id in node_subset:
                sub.add_edge(edge)
        return sub

    def connected_components(self) -> list[set[str]]:
        """Find connected components using Union-Find.

        Returns:
            List of sets, each containing member IDs of a component.
        """
        parent: dict[str, str] = {}
        rank: dict[str, int] = {}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1

        # Initialize
        for node in self._nodes:
            parent[node] = node
            rank[node] = 0

        # Union all edges
        for edge in self._edges.values():
            union(edge.source_id, edge.target_id)

        # Group by root
        components: dict[str, set[str]] = defaultdict(set)
        for node in self._nodes:
            root = find(node)
            components[root].add(node)

        return list(components.values())

    def size(self) -> tuple[int, int]:
        """Return (num_nodes, num_edges)."""
        return len(self._nodes), len(self._edges)

    def filter_edges(self, threshold: float) -> RelationshipGraph:
        """Return a new graph with only edges above threshold."""
        filtered = RelationshipGraph()
        for edge in self._edges.values():
            if edge.similarity >= threshold:
                filtered.add_edge(edge)
        return filtered

    def remove_nodes(self, nodes: set[str]) -> None:
        """Remove nodes and their incident edges from the graph."""
        for node in nodes:
            if node not in self._nodes:
                continue
            # Remove edges
            for neighbor in list(self._adjacency[node].keys()):
                pair = tuple(sorted([node, neighbor]))
                self._edges.pop(pair, None)
                self._adjacency[neighbor].pop(node, None)
                self._degrees[neighbor] = max(0, self._degrees[neighbor] - 1)
            # Remove node
            self._adjacency.pop(node, None)
            self._degrees.pop(node, None)
            self._nodes.discard(node)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node: str) -> bool:
        return node in self._nodes