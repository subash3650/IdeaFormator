"""Connected Components clustering provider (Baseline)."""

from __future__ import annotations

from collections import defaultdict

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipGraph
from phase2.clustering.providers.base import ClusterProvider
from phase2.clustering.providers.registry import register_provider


@register_provider("connected_components")
class ConnectedComponentsProvider(ClusterProvider):
    """Connected Components clustering provider.

    Discovers isolated subgraphs where every node is reachable from any other
    node in the subgraph through some path of relationships.

    This provider owns the connected component discovery algorithm (DFS/BFS/Union-Find)
    so that RelationshipGraph remains purely a graph data structure.
    """

    def cluster(self, graph: RelationshipGraph, config: ClusteringConfig) -> list[list[str]]:
        """Run connected components discovery on the graph."""
        nodes = sorted(list(graph.nodes()))
        visited = set()
        components: list[list[str]] = []

        # Run BFS/DFS in deterministic sorted order to ensure determinism
        for start_node in nodes:
            if start_node in visited:
                continue

            component = []
            queue = [start_node]
            visited.add(start_node)

            # Standard BFS
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                component.append(node)

                # Fetch neighbors and sort them deterministically
                neighbors = graph.neighbors(node)
                for neighbor in sorted(list(neighbors.keys())):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(sorted(component))

        return components

    @property
    def name(self) -> str:
        return "connected_components"

    @property
    def version(self) -> str:
        return "1.0"
