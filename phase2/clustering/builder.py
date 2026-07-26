"""ClusterBuilder — constructs immutable SemanticCluster objects."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipGraph
from phase2.clustering.providers.base import ClusterProvider
from phase2.clustering.schema import ClusterType, SemanticCluster


class ClusterBuilder:
    """Constructs immutable SemanticCluster objects with deterministic IDs."""

    def __init__(self, config: ClusteringConfig, provider: ClusterProvider) -> None:
        self._config = config
        self._provider = provider

    def build(
        self,
        member_ids: set[str] | list[str],
        graph: RelationshipGraph,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticCluster:
        """Build an immutable SemanticCluster from a set of member IDs.

        This handles:
            - Deterministic representative selection
            - Deterministic cluster ID generation
            - Assigning basic cluster metrics and metadata
        """
        members_list = sorted(list(set(member_ids)))
        member_count = len(members_list)

        # 1. Deterministic representative selection
        representative_id = self._select_representative(members_list, graph)

        # 2. Compute basic graph-based attributes for this cluster
        relationship_count = 0
        total_similarity = 0.0
        possible_relationships = member_count * (member_count - 1) / 2 if member_count > 1 else 1

        for i, u in enumerate(members_list):
            for v in members_list[i + 1 :]:
                weight = graph.edge_weight(u, v)
                if weight is not None:
                    relationship_count += 1
                    total_similarity += weight

        avg_similarity = total_similarity / max(relationship_count, 1)
        density = relationship_count / possible_relationships

        # 3. Generate deterministic cluster ID
        cluster_id = self._make_id(members_list)

        # 4. Construct the SemanticCluster
        return SemanticCluster(
            cluster_id=cluster_id,
            representative_id=representative_id,
            member_ids=tuple(members_list),
            member_count=member_count,
            relationship_count=relationship_count,
            average_similarity=avg_similarity,
            density=density,
            quality_score=0.0,  # Computed later by ClusterEvaluator
            cluster_type=ClusterType.NORMAL,  # Evaluator will mark LOW_QUALITY if needed
            provider=self._provider.name,
            provider_version=self._provider.version,
            algorithm=self._provider.name,
            metadata=metadata or {},
            version=self._config.version,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _select_representative(self, members: list[str], graph: RelationshipGraph) -> str:
        """Select the representative node for the cluster.

        Ranking strategy (deterministic, no randomness):
            1. Highest weighted degree (sum of internal edge weights)
            2. Highest average similarity to other members
            3. Lexicographically smallest ID
        """
        if not members:
            raise ValueError("Cannot select representative from an empty member list")
        if len(members) == 1:
            return members[0]

        # Compute ranking metrics for each candidate
        candidates: list[tuple[str, float, float]] = []
        for u in members:
            weighted_degree = 0.0
            similarities_sum = 0.0
            for v in members:
                if u == v:
                    continue
                weight = graph.edge_weight(u, v)
                if weight is not None:
                    weighted_degree += weight
                    similarities_sum += weight

            avg_similarity = similarities_sum / (len(members) - 1)
            candidates.append((u, weighted_degree, avg_similarity))

        # Sort candidates:
        # - weighted_degree descending (-x[1])
        # - avg_similarity descending (-x[2])
        # - ID ascending (x[0])
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (-x[1], -x[2], x[0]),
        )
        return sorted_candidates[0][0]

    def _make_id(self, sorted_members: list[str]) -> str:
        """Generate a deterministic cluster ID.

        SHA256(sorted(member_ids) | provider | provider_version | algorithm)
        Does NOT include quality weights, thresholds, configuration, or evaluation metrics.
        """
        raw = "|".join(sorted_members)
        raw += f"|{self._provider.name}|{self._provider.version}|{self._provider.name}"
        return hashlib.sha256(raw.encode()).hexdigest()
