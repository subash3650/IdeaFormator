"""ClusterSearcher — in-memory search over generated clusters."""

from __future__ import annotations

from collections import defaultdict

from phase2.clustering.schema import SemanticCluster


class ClusterSearcher:
    """Search interface for semantic clusters.

    Provides find_cluster, find_by_member, and find_by_representative.
    Independent from storage — operates on in-memory cluster lists.
    """

    def __init__(self, clusters: list[SemanticCluster]) -> None:
        self._clusters = clusters
        self._index: dict[str, SemanticCluster] = {}
        self._by_member: dict[str, list[str]] = defaultdict(list)
        self._by_representative: dict[str, str] = {}

        self._build_index()

    def _build_index(self) -> None:
        """Build lookup indexes from the cluster list."""
        for c in self._clusters:
            self._index[c.cluster_id] = c
            for member_id in c.member_ids:
                self._by_member[member_id].append(c.cluster_id)
            self._by_representative[c.representative_id] = c.cluster_id

    def find_cluster(self, cluster_id: str) -> SemanticCluster | None:
        """Find a cluster by its cluster ID."""
        return self._index.get(cluster_id)

    def find_by_member(self, member_id: str) -> list[SemanticCluster]:
        """Find all clusters that contain the given member ID."""
        cluster_ids = self._by_member.get(member_id, [])
        return [self._index[cid] for cid in cluster_ids if cid in self._index]

    def find_by_representative(self, representative_id: str) -> SemanticCluster | None:
        """Find the cluster whose representative is the given ID."""
        cid = self._by_representative.get(representative_id)
        if cid is not None:
            return self._index.get(cid)
        return None

    @property
    def total_clusters(self) -> int:
        return len(self._clusters)
