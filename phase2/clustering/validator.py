"""ClusterValidator — performs structured validation on generated cluster sets."""

from __future__ import annotations

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipGraph
from phase2.clustering.schema import SemanticCluster, ValidationIssue, ValidationResult


class ClusterValidator:
    """Validates the structure, integrity, and constraints of a semantic cluster set."""

    def __init__(self, config: ClusteringConfig) -> None:
        self._config = config

    def validate(self, clusters: list[SemanticCluster], graph: RelationshipGraph) -> ValidationResult:
        """Validate all generated clusters against integrity and structural rules.

        Returns:
            ValidationResult containing a list of ValidationIssue diagnostics.
        """
        issues: list[ValidationIssue] = []
        all_graph_nodes = graph.nodes()

        clusters_checked = len(clusters)
        members_checked = 0

        # Track members across all clusters to find duplicates or orphans
        seen_members_globally: dict[str, str] = {}  # member_id -> cluster_id

        for cluster in clusters:
            cid = cluster.cluster_id
            members = cluster.member_ids
            members_set = set(members)
            members_checked += len(members)

            # Rule 1: Size constraints
            if len(members) < self._config.minimum_cluster_size:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="MIN_SIZE_VIOLATION",
                        message=(
                            f"Cluster has {len(members)} members, which is below "
                            f"the minimum size of {self._config.minimum_cluster_size}."
                        ),
                        cluster_id=cid,
                    )
                )

            if len(members) > self._config.maximum_cluster_size:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="MAX_SIZE_VIOLATION",
                        message=(
                            f"Cluster has {len(members)} members, which exceeds "
                            f"the maximum size of {self._config.maximum_cluster_size}."
                        ),
                        cluster_id=cid,
                    )
                )

            # Rule 2: No duplicates within the cluster
            if len(members_set) != len(members):
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="DUPLICATE_MEMBERS",
                        message="Cluster contains duplicate member IDs.",
                        cluster_id=cid,
                    )
                )

            # Rule 3: Representative exists and belongs to the cluster
            if not cluster.representative_id:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="MISSING_REPRESENTATIVE",
                        message="Cluster has no assigned representative.",
                        cluster_id=cid,
                    )
                )
            elif cluster.representative_id not in members_set:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="REPRESENTATIVE_NOT_IN_CLUSTER",
                        message=(
                            f"Representative ID '{cluster.representative_id}' "
                            f"does not belong to the cluster members list."
                        ),
                        cluster_id=cid,
                        member_id=cluster.representative_id,
                    )
                )

            # Rule 4: Deterministic ordering (sorted member_ids)
            if list(members) != sorted(list(members)):
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="UNSORTED_MEMBERS",
                        message="Cluster member IDs are not deterministically sorted.",
                        cluster_id=cid,
                    )
                )

            # Rule 5: Orphan IDs and global existence check
            for member_id in members:
                if member_id not in all_graph_nodes:
                    issues.append(
                        ValidationIssue(
                            severity="ERROR",
                            code="ORPHAN_MEMBER_ID",
                            message=(
                                f"Member ID '{member_id}' does not exist as a node "
                                f"in the relationship graph."
                            ),
                            cluster_id=cid,
                            member_id=member_id,
                        )
                    )

                # Overlap / duplicate check globally (assuming non-overlapping clustering partition)
                if member_id in seen_members_globally:
                    other_cid = seen_members_globally[member_id]
                    # This might be a soft warning or error depending on algorithm.
                    # For connected components/hierarchical, partitions should be non-overlapping.
                    issues.append(
                        ValidationIssue(
                            severity="WARN",
                            code="OVERLAPPING_MEMBER",
                            message=(
                                f"Member ID '{member_id}' belongs to multiple clusters "
                                f"({cid} and {other_cid})."
                            ),
                            cluster_id=cid,
                            member_id=member_id,
                        )
                    )
                else:
                    seen_members_globally[member_id] = cid

            # Rule 6: Check that at least some relationships exist within the cluster (unless size = 1)
            if len(members) > 1 and cluster.relationship_count == 0:
                issues.append(
                    ValidationIssue(
                        severity="WARN",
                        code="DISCONNECTED_CLUSTER",
                        message="Cluster has multiple members but zero internal relationships.",
                        cluster_id=cid,
                    )
                )

        # Overall diagnostics
        has_errors = any(issue.severity == "ERROR" for issue in issues)

        return ValidationResult(
            valid=not has_errors,
            issues=issues,
            clusters_checked=clusters_checked,
            members_checked=members_checked,
        )
