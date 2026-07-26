"""Metrics and statistics for semantic cluster sets."""

from __future__ import annotations

import numpy as np

from phase2.clustering.schema import ClusterStats, SemanticCluster


def compute_cluster_stats(clusters: list[SemanticCluster]) -> ClusterStats:
    """Compute aggregate statistics over a list of SemanticCluster objects.

    Returns:
        A fully populated ClusterStats object.
    """
    total_clusters = len(clusters)
    if total_clusters == 0:
        return ClusterStats(
            total_clusters=0,
            total_members=0,
            total_relationships=0,
            average_cluster_size=0.0,
            average_density=0.0,
            average_quality=0.0,
            cluster_size_min=0,
            cluster_size_max=0,
            cluster_size_median=0.0,
            quality_distribution={},
            cluster_type_counts={},
        )

    sizes = [c.member_count for c in clusters]
    densities = [c.density for c in clusters]
    qualities = [c.quality_score for c in clusters]

    total_members = sum(len(c.member_ids) for c in clusters)
    total_relationships = sum(c.relationship_count for c in clusters)

    average_cluster_size = float(np.mean(sizes))
    average_density = float(np.mean(densities))
    average_quality = float(np.mean(qualities))

    cluster_size_min = int(np.min(sizes))
    cluster_size_max = int(np.max(sizes))
    cluster_size_median = float(np.median(sizes))

    # Quality distribution in 10 equal bins from 0.0 to 1.0
    quality_bins = [round(i * 0.1, 1) for i in range(11)]
    quality_counts = {f"{quality_bins[i]:.1f}-{quality_bins[i+1]:.1f}": 0 for i in range(10)}

    for q in qualities:
        for i in range(10):
            if quality_bins[i] <= q < quality_bins[i+1]:
                quality_counts[f"{quality_bins[i]:.1f}-{quality_bins[i+1]:.1f}"] += 1
                break
        else:
            # Handle boundary condition of exactly 1.0
            if q >= 1.0:
                quality_counts["0.9-1.0"] += 1

    # Filter out empty bins for clean output
    quality_dist_cleaned = {k: v for k, v in quality_counts.items() if v > 0}

    # Cluster type distribution
    cluster_type_counts: dict[str, int] = {}
    for c in clusters:
        ctype = c.cluster_type.value
        cluster_type_counts[ctype] = cluster_type_counts.get(ctype, 0) + 1

    return ClusterStats(
        total_clusters=total_clusters,
        total_members=total_members,
        total_relationships=total_relationships,
        average_cluster_size=average_cluster_size,
        average_density=average_density,
        average_quality=average_quality,
        cluster_size_min=cluster_size_min,
        cluster_size_max=cluster_size_max,
        cluster_size_median=cluster_size_median,
        quality_distribution=quality_dist_cleaned,
        cluster_type_counts=cluster_type_counts,
    )
