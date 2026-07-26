"""CustomGraph — lightweight in-memory graph implementation.

No external dependencies. Uses dict-based adjacency indexes for O(1) lookups.
"""

from __future__ import annotations

from collections import deque

from phase2.knowledge_graph.adjacency import AdjacencyIndex, ReverseAdjacencyIndex
from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphMetadata, GraphNode, NodeType


class CustomGraph(GraphInterface):
    """Lightweight in-memory directed graph.

    Stores nodes and edges in dicts keyed by ID.
    Maintains typed indexes for O(1) type-filtered lookups.
    Uses adjacency indexes for O(1) neighbor/predecessor queries.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._type_index: dict[str, set[str]] = {}
        self._edge_type_index: dict[str, set[str]] = {}
        self._adjacency = AdjacencyIndex()
        self._reverse_adjacency = ReverseAdjacencyIndex()

    # ── Mutation ──────────────────────────────────────────────────────

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.node_id] = node
        self._type_index.setdefault(node.node_type.value, set()).add(node.node_id)

    def add_edge(self, edge: GraphEdge) -> None:
        self._edges[edge.edge_id] = edge
        self._edge_type_index.setdefault(edge.edge_type.value, set()).add(edge.edge_id)
        self._adjacency.add(edge)
        self._reverse_adjacency.add(edge)

    def remove_node(self, node_id: str) -> None:
        node = self._nodes.pop(node_id, None)
        if node is None:
            return
        node_type_val = node.node_type.value
        type_set = self._type_index.get(node_type_val)
        if type_set:
            type_set.discard(node_id)
        self._adjacency.remove_node(node_id)
        self._reverse_adjacency.remove_node(node_id)
        edges_to_remove = [eid for eid, e in self._edges.items() if e.source_node_id == node_id or e.target_node_id == node_id]
        for eid in edges_to_remove:
            self.remove_edge(eid)

    def remove_edge(self, edge_id: str) -> None:
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return
        edge_type_val = edge.edge_type.value
        type_set = self._edge_type_index.get(edge_type_val)
        if type_set:
            type_set.discard(edge_id)
        self._adjacency.remove(edge_id, edge)
        self._reverse_adjacency.remove(edge_id, edge)

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._type_index.clear()
        self._edge_type_index.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()

    # ── Accessors ─────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        return self._edges.get(edge_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self._edges

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    def edges(self) -> list[GraphEdge]:
        return list(self._edges.values())

    def nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        ids = self._type_index.get(node_type.value, set())
        return [self._nodes[nid] for nid in ids if nid in self._nodes]

    def edges_by_type(self, edge_type: EdgeType) -> list[GraphEdge]:
        ids = self._edge_type_index.get(edge_type.value, set())
        return [self._edges[eid] for eid in ids if eid in self._edges]

    def node_type_counts(self) -> dict[str, int]:
        return {t: len(ids) for t, ids in self._type_index.items()}

    def edge_type_counts(self) -> dict[str, int]:
        return {t: len(ids) for t, ids in self._edge_type_index.items()}

    # ── Traversal ─────────────────────────────────────────────────────

    def neighbors(self, node_id: str, edge_type: EdgeType | None = None, direction: str = "out") -> list[str]:
        if direction == "in":
            return self._reverse_adjacency.predecessors(node_id, edge_type)
        return self._adjacency.neighbors(node_id, edge_type)

    def predecessors(self, node_id: str, edge_type: EdgeType | None = None) -> list[str]:
        return self._reverse_adjacency.predecessors(node_id, edge_type)

    def successors(self, node_id: str, edge_type: EdgeType | None = None) -> list[str]:
        return self._adjacency.neighbors(node_id, edge_type)

    def out_degree(self, node_id: str) -> int:
        return self._adjacency.out_degree(node_id)

    def in_degree(self, node_id: str) -> int:
        return self._reverse_adjacency.in_degree(node_id)

    def degree(self, node_id: str) -> int:
        return self.out_degree(node_id) + self.in_degree(node_id)

    # ── Subgraph ──────────────────────────────────────────────────────

    def subgraph(self, node_ids: set[str]) -> CustomGraph:
        sub = CustomGraph()
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is not None:
                sub.add_node(node)
        for edge in self._edges.values():
            if edge.source_node_id in node_ids and edge.target_node_id in node_ids:
                sub.add_edge(edge)
        return sub

    # ── Metadata ──────────────────────────────────────────────────────

    def metadata(self, run_id: str) -> GraphMetadata:
        n_count = self.node_count()
        e_count = self.edge_count()
        ntc = self.node_type_counts()
        etc = self.edge_type_counts()
        comps = self._compute_components()
        largest = max((len(c) for c in comps), default=0)
        density = (2 * e_count) / (n_count * (n_count - 1)) if n_count > 1 else 0.0
        avg_conf = sum(n.confidence for n in self._nodes.values()) / max(n_count, 1)
        avg_deg = sum(self.degree(nid) for nid in self._nodes) / max(n_count, 1)
        orphan_count = sum(1 for nid in self._nodes if self.degree(nid) == 0)
        return GraphMetadata(
            graph_id=run_id,
            node_count=n_count,
            edge_count=e_count,
            node_type_counts=ntc,
            edge_type_counts=etc,
            connected_components=len(comps),
            largest_component_size=largest,
            density=round(density, 6),
            avg_confidence=round(avg_conf, 6),
            avg_degree=round(avg_deg, 6),
            orphan_node_count=orphan_count,
            run_id=run_id,
            pipeline_version="",
            schema_version="",
        )

    def _compute_components(self) -> list[set[str]]:
        all_nodes = set(self._nodes.keys())
        visited: set[str] = set()
        components: list[set[str]] = []

        for node_id in all_nodes:
            if node_id in visited:
                continue
            component: set[str] = set()
            queue: deque[str] = deque([node_id])
            visited.add(node_id)
            while queue:
                current = queue.popleft()
                component.add(current)
                for nbr_id in self.neighbors(current):
                    if nbr_id not in visited:
                        visited.add(nbr_id)
                        queue.append(nbr_id)
                for pred_id in self.predecessors(current):
                    if pred_id not in visited:
                        visited.add(pred_id)
                        queue.append(pred_id)
            components.append(component)
        return components
