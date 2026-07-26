"""Pure-Python graph algorithms — no external dependencies."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phase2.knowledge_graph.graph_interface import GraphInterface


def bfs(graph: GraphInterface, start: str, max_depth: int | None = None) -> list[str]:
    """Breadth-first search, returning visited node IDs in order."""
    visited: list[str] = []
    queue: deque[tuple[str, int]] = deque()
    queue.append((start, 0))
    seen: set[str] = {start}
    while queue:
        node, depth = queue.popleft()
        visited.append(node)
        if max_depth is not None and depth >= max_depth:
            continue
        for neighbor in graph.neighbors(node):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
    return visited


def dfs(graph: GraphInterface, start: str, max_depth: int | None = None) -> list[str]:
    """Depth-first search, returning visited node IDs in order."""
    visited: list[str] = []
    stack: list[tuple[str, int]] = [(start, 0)]
    seen: set[str] = {start}
    while stack:
        node, depth = stack.pop()
        visited.append(node)
        if max_depth is not None and depth >= max_depth:
            continue
        for neighbor in reversed(graph.neighbors(node)):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append((neighbor, depth + 1))
    return visited


def connected_components(graph: GraphInterface) -> list[set[str]]:
    """Return connected components as sets of node IDs."""
    all_nodes = set(n.node_id for n in graph.nodes())
    visited: set[str] = set()
    components: list[set[str]] = []
    for node in all_nodes:
        if node in visited:
            continue
        component: set[str] = set()
        queue: deque[str] = deque([node])
        visited.add(node)
        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
            for predecessor in graph.predecessors(current):
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append(predecessor)
        components.append(component)
    return components


def strongly_connected_components(graph: GraphInterface) -> list[set[str]]:
    """Return strongly connected components using Tarjan's algorithm."""
    all_nodes = list(n.node_id for n in graph.nodes())
    index_counter = 0
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: set[str] = set()
    result: list[set[str]] = []

    def _strongconnect(node_id: str) -> None:
        nonlocal index_counter
        index[node_id] = index_counter
        lowlink[node_id] = index_counter
        index_counter += 1
        stack.append(node_id)
        on_stack.add(node_id)
        for successor in graph.neighbors(node_id):
            if successor not in index:
                _strongconnect(successor)
                lowlink[node_id] = min(lowlink[node_id], lowlink[successor])
            elif successor in on_stack:
                lowlink[node_id] = min(lowlink[node_id], index[successor])
        if lowlink[node_id] == index[node_id]:
            component: set[str] = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.add(w)
                if w == node_id:
                    break
            result.append(component)

    for node in all_nodes:
        if node not in index:
            _strongconnect(node)
    return result


def shortest_path(graph: GraphInterface, source: str, target: str) -> list[str] | None:
    """BFS-based shortest path between two nodes. Returns list of node IDs or None."""
    if source == target:
        return [source]
    queue: deque[list[str]] = deque([[source]])
    visited: set[str] = {source}
    while queue:
        path = queue.popleft()
        last = path[-1]
        for neighbor in graph.neighbors(last):
            if neighbor == target:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None


def shortest_paths(graph: GraphInterface, source: str, max_depth: int = 5) -> dict[str, list[str]]:
    """Return shortest paths from source to all reachable nodes within max_depth."""
    paths: dict[str, list[str]] = {}
    queue: deque[list[str]] = deque([[source]])
    visited: set[str] = {source}
    while queue:
        path = queue.popleft()
        last = path[-1]
        if len(path) - 1 > max_depth:
            continue
        if last != source:
            paths[last] = list(path)
        for neighbor in graph.neighbors(last):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return paths


def topological_sort(graph: GraphInterface) -> list[str] | None:
    """Kahn's algorithm. Returns ordered node IDs or None if cycle exists."""
    all_nodes = list(n.node_id for n in graph.nodes())
    in_degree: dict[str, int] = {n: 0 for n in all_nodes}
    for node in all_nodes:
        for _ in graph.neighbors(node):
            in_degree[node] = in_degree.get(node, 0)
    for node in all_nodes:
        for neighbor in graph.neighbors(node):
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    queue: deque[str] = deque([n for n in all_nodes if in_degree.get(n, 0) == 0])
    result: list[str] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.neighbors(node):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(result) != len(all_nodes):
        return None
    return result


def has_cycle(graph: GraphInterface) -> bool:
    """Detect directed cycles via DFS-based coloring."""
    all_nodes = list(n.node_id for n in graph.nodes())
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in all_nodes}

    def _dfs(node: str) -> bool:
        color[node] = GRAY
        for neighbor in graph.neighbors(node):
            if color.get(neighbor) == GRAY:
                return True
            if color.get(neighbor) == WHITE:
                if _dfs(neighbor):
                    return True
        color[node] = BLACK
        return False

    for node in all_nodes:
        if color[node] == WHITE:
            if _dfs(node):
                return True
    return False


def degree_centrality(graph: GraphInterface, top_k: int = 10) -> list[tuple[str, float]]:
    """Return top-k nodes by normalized degree centrality."""
    all_nodes = list(n.node_id for n in graph.nodes())
    n = len(all_nodes)
    if n <= 1:
        return [(node_id, 1.0) for node_id in all_nodes]
    max_degree = n - 1
    scores: list[tuple[str, float]] = []
    for node_id in all_nodes:
        deg = graph.degree(node_id)
        scores.append((node_id, deg / max_degree))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:top_k]


def betweenness_centrality(graph: GraphInterface, top_k: int = 10) -> list[tuple[str, float]]:
    """Approximate betweenness centrality via BFS on sample nodes.

    Uses all nodes for accuracy; for very large graphs, sampling would be added later.
    """
    all_nodes = list(n.node_id for n in graph.nodes())
    centrality: dict[str, float] = {n: 0.0 for n in all_nodes}

    for s in all_nodes:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {n: [] for n in all_nodes}
        sigma: dict[str, float] = {n: 0.0 for n in all_nodes}
        dist: dict[str, int | None] = {n: None for n in all_nodes}
        sigma[s] = 1.0
        dist[s] = 0
        queue: deque[str] = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph.neighbors(v):
                w_dist = dist[w]
                if w_dist is None:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                    sigma[w] = sigma[v]
                    predecessors[w].append(v)
                elif dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        delta: dict[str, float] = {n: 0.0 for n in all_nodes}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    n = len(all_nodes)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2))
        for node_id in centrality:
            centrality[node_id] *= scale

    scores = sorted(centrality.items(), key=lambda x: (-x[1], x[0]))
    return scores[:top_k]


def pagerank(
    graph: GraphInterface,
    damping: float = 0.85,
    iterations: int = 100,
    top_k: int = 10,
    tolerance: float = 1e-6,
) -> list[tuple[str, float]]:
    """PageRank implementation. Returns top-k nodes by score."""
    all_nodes = list(n.node_id for n in graph.nodes())
    n = len(all_nodes)
    if n == 0:
        return []
    if n == 1:
        return [(all_nodes[0], 1.0)]

    node_index = {node_id: i for i, node_id in enumerate(all_nodes)}
    out_degree_counts = [graph.out_degree(node_id) for node_id in all_nodes]

    rank = [1.0 / n] * n
    for _ in range(iterations):
        prev = rank[:]
        for i, node_id in enumerate(all_nodes):
            incoming_sum = 0.0
            for pred in graph.predecessors(node_id):
                j = node_index[pred]
                if out_degree_counts[j] > 0:
                    incoming_sum += prev[j] / out_degree_counts[j]
            rank[i] = (1.0 - damping) / n + damping * incoming_sum
        diff = sum(abs(rank[i] - prev[i]) for i in range(n))
        if diff < tolerance:
            break

    scores = [(all_nodes[i], rank[i]) for i in range(n)]
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:top_k]


def common_neighbors(graph: GraphInterface, node_a: str, node_b: str) -> list[str]:
    """Return node IDs that are neighbors of both node_a and node_b."""
    neighbors_a = set(graph.neighbors(node_a))
    neighbors_a.update(graph.predecessors(node_a))
    neighbors_b = set(graph.neighbors(node_b))
    neighbors_b.update(graph.predecessors(node_b))
    return sorted(neighbors_a & neighbors_b)
