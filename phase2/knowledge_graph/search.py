"""GraphSearch — comprehensive search and traversal API for knowledge graph."""

from __future__ import annotations

from phase2.knowledge_graph.algorithms import common_neighbors, connected_components, degree_centrality, shortest_path
from phase2.knowledge_graph.graph import CustomGraph
from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.schema import EdgeType, GraphEdge, GraphNode, NodeType


class GraphSearch:
    """Search and traversal API for the knowledge graph."""

    def __init__(self, graph: GraphInterface) -> None:
        self._graph = graph

    # ── Node queries ──────────────────────────────────────────────────

    def find_node(self, node_id: str) -> GraphNode | None:
        return self._graph.get_node(node_id)

    def find_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        return self._graph.nodes_by_type(node_type)

    def find_by_label(self, query: str, fuzzy: bool = True) -> list[GraphNode]:
        query_lower = query.lower()
        results: list[GraphNode] = []
        for node in self._graph.nodes():
            if fuzzy:
                if query_lower in node.label.lower():
                    results.append(node)
            else:
                if node.label == query:
                    results.append(node)
        return results

    def find_by_property(self, key: str, value: object) -> list[GraphNode]:
        results: list[GraphNode] = []
        for node in self._graph.nodes():
            if key in node.properties and node.properties[key] == value:
                results.append(node)
        return results

    # ── Domain-specific queries ───────────────────────────────────────

    def find_documents(self, **filters: object) -> list[GraphNode]:
        return self._find_by_type_with_filters(NodeType.DOCUMENT, filters)

    def find_observations(self, **filters: object) -> list[GraphNode]:
        return self._find_by_type_with_filters(NodeType.OBSERVATION, filters)

    def find_entities(self, **filters: object) -> list[GraphNode]:
        return self._find_by_type_with_filters(NodeType.ENTITY, filters)

    def find_sources(self, **filters: object) -> list[GraphNode]:
        return self._find_by_type_with_filters(NodeType.SOURCE, filters)

    def find_companies(self, **filters: object) -> list[GraphNode]:
        return self._find_by_type_with_filters(NodeType.COMPANY, filters)

    def find_products(self, **filters: object) -> list[GraphNode]:
        return self._find_by_type_with_filters(NodeType.PRODUCT, filters)

    def find_problem_signals(self, **filters: object) -> list[GraphNode]:
        return self._find_by_type_with_filters(NodeType.PROBLEM_SIGNAL, filters)

    def find_clusters(self, **filters: object) -> list[GraphNode]:
        return self._find_by_type_with_filters(NodeType.CLUSTER, filters)

    def find_evidence(self, **filters: object) -> list[GraphNode]:
        return self._find_by_type_with_filters(NodeType.EVIDENCE, filters)

    def find_similar(self, node_id: str, top_k: int = 10) -> list[GraphNode]:
        node = self._graph.get_node(node_id)
        if node is None:
            return []
        scored_nodes: list[tuple[GraphNode, float]] = []
        for e in self._graph.edges_by_type(EdgeType.SIMILAR_TO):
            if e.source_node_id == node_id:
                neighbor = self._graph.get_node(e.target_node_id)
                if neighbor is not None:
                    scored_nodes.append((neighbor, e.weight))
            elif e.target_node_id == node_id:
                neighbor = self._graph.get_node(e.source_node_id)
                if neighbor is not None:
                    scored_nodes.append((neighbor, e.weight))
        scored_nodes.sort(key=lambda x: (-x[1], x[0].node_id))
        return [n for n, _ in scored_nodes[:top_k]]

    # ── Neighborhood ──────────────────────────────────────────────────

    def find_neighbors(self, node_id: str, depth: int = 1, edge_type: EdgeType | None = None) -> list[GraphNode]:
        if depth <= 0:
            node = self._graph.get_node(node_id)
            return [node] if node else []
        from collections import deque
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        result_nodes: list[GraphNode] = []
        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue
            for nid in self._graph.neighbors(current, edge_type=edge_type):
                if nid not in visited:
                    visited.add(nid)
                    neighbor = self._graph.get_node(nid)
                    if neighbor:
                        result_nodes.append(neighbor)
                    queue.append((nid, d + 1))
        return result_nodes

    def find_common_neighbors(self, node_a: str, node_b: str) -> list[GraphNode]:
        nids = common_neighbors(self._graph, node_a, node_b)
        return [n for n in self._graph.nodes() if n.node_id in nids]

    def find_predecessors(self, node_id: str, depth: int = 1, edge_type: EdgeType | None = None) -> list[GraphNode]:
        if depth <= 0:
            node = self._graph.get_node(node_id)
            return [node] if node else []
        from collections import deque
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        result_nodes: list[GraphNode] = []
        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue
            for pred_id in self._graph.predecessors(current, edge_type=edge_type):
                if pred_id not in visited:
                    visited.add(pred_id)
                    pred = self._graph.get_node(pred_id)
                    if pred:
                        result_nodes.append(pred)
                    queue.append((pred_id, d + 1))
        return result_nodes

    def find_successors(self, node_id: str, depth: int = 1, edge_type: EdgeType | None = None) -> list[GraphNode]:
        return self.find_neighbors(node_id, depth=depth, edge_type=edge_type)

    # ── Path queries ──────────────────────────────────────────────────

    def find_path(self, source: str, target: str, max_depth: int = 5) -> list[str] | None:
        path = shortest_path(self._graph, source, target)
        if path and len(path) - 1 > max_depth:
            return None
        return path

    def find_all_paths(self, source: str, target: str, max_depth: int = 5) -> list[list[str]]:
        all_paths: list[list[str]] = []
        from collections import deque
        queue: deque[list[str]] = deque([[source]])
        while queue:
            path = queue.popleft()
            last = path[-1]
            if len(path) - 1 > max_depth:
                continue
            if last == target and len(path) > 1:
                all_paths.append(path)
                continue
            for nid in self._graph.neighbors(last):
                if nid not in path:
                    queue.append(path + [nid])
        return all_paths

    # ── Cluster / community ───────────────────────────────────────────

    def find_cluster(self, cluster_id: str) -> list[GraphNode]:
        members: list[GraphNode] = []
        for edge in self._graph.edges():
            if edge.edge_type == EdgeType.MEMBER_OF_CLUSTER and edge.target_node_id == cluster_id:
                member = self._graph.get_node(edge.source_node_id)
                if member:
                    members.append(member)
            elif edge.edge_type == EdgeType.BELONGS_TO and edge.target_node_id == cluster_id:
                member = self._graph.get_node(edge.source_node_id)
                if member:
                    members.append(member)
        return members

    def find_communities(self) -> list[set[str]]:
        return connected_components(self._graph)

    # ── Ranking ───────────────────────────────────────────────────────

    def find_central_nodes(self, metric: str = "degree", top_k: int = 10) -> list[GraphNode]:
        if metric == "degree":
            scores = degree_centrality(self._graph, top_k=top_k)
            nid_set = set(nid for nid, _ in scores)
            return [n for n in self._graph.nodes() if n.node_id in nid_set]
        return []

    def find_by_confidence(self, min_confidence: float) -> list[GraphNode]:
        return [n for n in self._graph.nodes() if n.confidence >= min_confidence]

    def find_by_weight(self, min_weight: float) -> list[GraphEdge]:
        return [e for e in self._graph.edges() if e.weight >= min_weight]

    # ── Helpers ───────────────────────────────────────────────────────

    def _find_by_type_with_filters(self, node_type: NodeType, filters: dict[str, object]) -> list[GraphNode]:
        nodes = self._graph.nodes_by_type(node_type)
        if not filters:
            return nodes
        results: list[GraphNode] = []
        for node in nodes:
            match = True
            for key, value in filters.items():
                if key in node.properties:
                    if node.properties[key] != value:
                        match = False
                        break
                elif key in node.metadata:
                    if node.metadata[key] != value:
                        match = False
                        break
                elif key in node.attributes:
                    if node.attributes[key] != value:
                        match = False
                        break
            if match:
                results.append(node)
        return results
