"""ClusterEvaluator — evaluates cluster quality and computes multidimensional metrics."""

from __future__ import annotations

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipGraph
from phase2.clustering.schema import ClusterMetrics, ClusterType, SemanticCluster


class ClusterEvaluator:
    """Evaluates semantic clusters, computing quality scores and metrics.

    The quality score is a configurable weighted combination of:
        - Internal Cohesion (average internal edge similarity)
        - Density (actual / possible internal edges)
        - External Separation (1.0 - average external edge similarity)
        - Connectivity (average internal degree / max possible internal degree)

    This component is completely independent of the cluster providers.
    """

    def __init__(self, config: ClusteringConfig) -> None:
        self._config = config

    def evaluate(self, cluster: SemanticCluster, graph: RelationshipGraph) -> ClusterMetrics:
        """Compute comprehensive metrics and quality score for a single cluster.

        Args:
            cluster: The SemanticCluster to evaluate.
            graph: The full RelationshipGraph containing all nodes and edges.
        """
        members = set(cluster.member_ids)
        member_count = len(members)

        # 1. Edge calculations (Internal vs. External)
        internal_edges_count = 0
        external_edges_count = 0
        total_internal_similarity = 0.0
        total_external_similarity = 0.0

        internal_degrees: dict[str, int] = {m: 0 for m in members}
        total_degrees: dict[str, int] = {m: 0 for m in members}

        # Scan all edges in the graph once to count internal vs external
        for edge in graph.edges():
            u, v = edge.source_id, edge.target_id
            u_in = u in members
            v_in = v in members

            if u_in and v_in:
                # Internal edge
                internal_edges_count += 1
                total_internal_similarity += edge.similarity
                internal_degrees[u] += 1
                internal_degrees[v] += 1
                total_degrees[u] += 1
                total_degrees[v] += 1
            elif u_in or v_in:
                # External edge
                external_edges_count += 1
                total_external_similarity += edge.similarity
                node_in = u if u_in else v
                total_degrees[node_in] += 1

        # 2. Basic degree statistics (Internal degree within the cluster)
        deg_values = list(internal_degrees.values())
        avg_degree = sum(deg_values) / max(member_count, 1)
        max_degree = max(deg_values) if deg_values else 0
        min_degree = min(deg_values) if deg_values else 0

        # Total edge count incident to any node in cluster
        edge_count = internal_edges_count + external_edges_count

        # 3. Core Quality Dimensions
        # Cohesion: Average similarity of internal edges (or 0.0 if none)
        internal_cohesion = total_internal_similarity / max(internal_edges_count, 1)

        # Density: actual / possible internal edges
        possible_internal = member_count * (member_count - 1) / 2 if member_count > 1 else 1
        density = internal_edges_count / possible_internal

        # Separation: 1.0 - average similarity of external edges (or 1.0 if none)
        avg_external_sim = total_external_similarity / max(external_edges_count, 1)
        external_separation = 1.0 - avg_external_sim if external_edges_count > 0 else 1.0

        # Connectivity: average internal degree normalized by possible connections
        connectivity = avg_degree / max(member_count - 1, 1) if member_count > 1 else 1.0

        # 4. Composite Quality Score
        weights = self._config.quality_weights.normalize()
        quality_score = (
            weights.cohesion * internal_cohesion
            + weights.density * density
            + weights.separation * external_separation
            + weights.connectivity * connectivity
        )

        # Ensure bounded
        quality_score = max(0.0, min(1.0, quality_score))

        return ClusterMetrics(
            cluster_id=cluster.cluster_id,
            member_count=member_count,
            relationship_count=internal_edges_count,
            average_similarity=internal_cohesion,
            density=density,
            average_degree=avg_degree,
            max_degree=max_degree,
            min_degree=min_degree,
            edge_count=edge_count,
            internal_edge_count=internal_edges_count,
            external_edge_count=external_edges_count,
            internal_cohesion=internal_cohesion,
            external_separation=external_separation,
            connectivity=connectivity,
            quality_score=quality_score,
        )

    def evaluate_and_update(self, cluster: SemanticCluster, graph: RelationshipGraph) -> SemanticCluster:
        """Evaluate a cluster and return a new updated SemanticCluster copy with metrics."""
        metrics = self.evaluate(cluster, graph)

        # Determine cluster type based on quality threshold
        cluster_type = ClusterType.NORMAL
        if metrics.quality_score < self._config.quality_threshold:
            cluster_type = ClusterType.LOW_QUALITY

        # Re-build cluster with updated quality scores
        return SemanticCluster(
            cluster_id=cluster.cluster_id,
            representative_id=cluster.representative_id,
            member_ids=cluster.member_ids,
            member_count=cluster.member_count,
            relationship_count=metrics.relationship_count,
            average_similarity=metrics.average_similarity,
            density=metrics.density,
            quality_score=metrics.quality_score,
            cluster_type=cluster_type,
            provider=cluster.provider,
            provider_version=cluster.provider_version,
            algorithm=cluster.algorithm,
            metadata=cluster.metadata,
            version=cluster.version,
            created_at=cluster.created_at,
        )
