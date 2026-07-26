"""Hierarchical clustering provider (Agglomerative with Average Linkage)."""

from __future__ import annotations

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipGraph
from phase2.clustering.providers.base import ClusterProvider
from phase2.clustering.providers.registry import register_provider


@register_provider("hierarchical")
class HierarchicalProvider(ClusterProvider):
    """Hierarchical agglomerative clustering provider.

    Groups concepts bottom-up using Average Linkage similarity and a
    deterministic tie-breaking strategy.

    To ensure high performance and avoid quadratic complexity over the entire
    dataset, it first splits the graph into connected components, then applies
    agglomerative clustering on each component independently.
    """

    def cluster(self, graph: RelationshipGraph, config: ClusteringConfig) -> list[list[str]]:
        """Cluster the graph hierarchically."""
        # Step 1: Pre-filter graph edges using the relationship threshold to find base components
        filtered_graph = graph.filter_edges(config.relationship_threshold)
        components = filtered_graph.connected_components()

        all_clusters: list[list[str]] = []

        # Step 2: Apply bottom-up agglomerative clustering on each component
        for component in components:
            if len(component) <= 1:
                all_clusters.append(list(component))
                continue

            # Run agglomerative clustering on this component
            component_clusters = self._agglomerative_cluster(
                list(component), graph, config.relationship_threshold
            )
            all_clusters.extend(component_clusters)

        # Sort the output clusters and members for determinism
        return [sorted(c) for c in sorted(all_clusters, key=lambda x: sorted(x))]

    def _agglomerative_cluster(
        self, nodes: list[str], graph: RelationshipGraph, threshold: float
    ) -> list[list[str]]:
        """Run average-linkage bottom-up clustering on a subset of nodes."""
        # Initialize each node as its own cluster (tuple of member IDs)
        clusters = [tuple([node]) for node in nodes]

        while len(clusters) > 1:
            best_pair = None
            best_sim = -1.0

            # Find the pair of clusters with the highest average similarity
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    c1, c2 = clusters[i], clusters[j]
                    sim = self._average_similarity(c1, c2, graph)

                    # Deterministic tie-breaking:
                    # Choose highest similarity, break ties lexicographically by member IDs
                    is_better = False
                    if sim > best_sim:
                        is_better = True
                    elif abs(sim - best_sim) < 1e-9:
                        # Tie break! Construct deterministic sorted string representation for pairs
                        current_pair = tuple(sorted([c1, c2], key=lambda x: sorted(x)))
                        existing_pair = tuple(sorted([best_pair[0], best_pair[1]], key=lambda x: sorted(x)))
                        if current_pair < existing_pair:
                            is_better = True

                    if is_better:
                        best_sim = sim
                        best_pair = (c1, c2)

            # Stopping condition: if highest average similarity is below the threshold, stop merging
            if best_pair is None or best_sim < threshold:
                break

            # Merge the best pair
            c1, c2 = best_pair
            clusters.remove(c1)
            clusters.remove(c2)
            merged = tuple(sorted(list(c1) + list(c2)))
            clusters.append(merged)

        return [list(c) for c in clusters]

    def _average_similarity(self, c1: tuple[str, ...], c2: tuple[str, ...], graph: RelationshipGraph) -> float:
        """Compute the average similarity between two clusters."""
        total_sim = 0.0
        for u in c1:
            for v in c2:
                sim = graph.edge_weight(u, v)
                if sim is not None:
                    total_sim += sim
        return total_sim / (len(c1) * len(c2))

    @property
    def name(self) -> str:
        return "hierarchical"

    @property
    def version(self) -> str:
        return "1.0"
