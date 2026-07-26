"""Export manifest, quality report, and JSON report for cluster generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from phase2.clustering.metrics import compute_cluster_stats
from phase2.clustering.schema import (
    ClusterManifest,
    ClusterReport,
    SemanticCluster,
)


def write_manifest(
    manifest: ClusterManifest,
    output_dir: Path,
) -> Path:
    """Write the cluster manifest as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cluster_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2, default=str)
    return path


def write_json_report(
    report: ClusterReport,
    output_dir: Path,
) -> Path:
    """Write the cluster JSON report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cluster_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
    return path


def generate_text_report(
    report: ClusterReport,
    output_dir: Path,
) -> Path:
    """Generate a human-readable text report."""
    lines = [
        "=" * 60,
        "Semantic Cluster Quality Report",
        "=" * 60,
        f"Generated: {report.generated_at}",
        f"Provider:  {report.provider}",
        f"Algorithm: {report.algorithm}",
        "",
        "--- Overview ---",
        f"Total clusters      : {report.total_clusters}",
        f"Total members       : {report.total_members}",
        f"Total relationships : {report.total_relationships}",
        f"Average cluster size: {report.average_cluster_size:.2f}",
        f"Overall density     : {report.cluster_density:.6f}",
        "",
        "--- Size Distribution ---",
        f"Smallest cluster: {report.cluster_size_min if hasattr(report, 'cluster_size_min') else 'N/A'}",
        f"Largest cluster : {report.cluster_size_max if hasattr(report, 'cluster_size_max') else 'N/A'}",
    ]

    if report.cluster_size_distribution:
        lines.append("")
        lines.append("Cluster Size Distribution:")
        for size, count in sorted(report.cluster_size_distribution.items()):
            lines.append(f"  Size {size}: {count} cluster(s)")

    lines.extend([
        "",
        "--- Quality ---",
        f"Average quality score: {report.average_quality if hasattr(report, 'average_quality') else 'N/A'}",
        f"Low quality clusters: {report.low_quality_count}",
        "",
        "--- Quality Distribution ---",
    ])
    for bucket, count in sorted(report.quality_distribution.items()):
        lines.append(f"  {bucket}: {count}")

    lines.extend([
        "",
        f"Singleton count      : {report.singleton_count}",
        f"Orphan concept count : {report.orphan_concept_count}",
        "",
        "--- Top Representative IDs ---",
    ])
    for rid in report.top_representative_ids[:10]:
        lines.append(f"  {rid}")

    if report.largest_clusters:
        lines.extend([
            "",
            "--- Largest Clusters ---",
        ])
        for cl in report.largest_clusters[:5]:
            lines.append(f"  {cl.get('cluster_id', '')[:16]}... | size={cl.get('size', 0)} | repr={cl.get('representative_id', '')[:16]}")

    if report.smallest_clusters:
        lines.extend([
            "",
            "--- Smallest Clusters ---",
        ])
        for cl in report.smallest_clusters[:5]:
            lines.append(f"  {cl.get('cluster_id', '')[:16]}... | size={cl.get('size', 0)} | repr={cl.get('representative_id', '')[:16]}")

    lines.extend([
        "",
        f"Elapsed time: {report.elapsed_seconds:.2f}s",
        "=" * 60,
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cluster_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def build_cluster_report(
    clusters: list[SemanticCluster],
    elapsed_seconds: float,
    provider: str,
    algorithm: str,
    relationship_count: int = 0,
    orphan_concept_count: int = 0,
) -> ClusterReport:
    """Build a ClusterReport from a list of evaluated clusters."""
    stats = compute_cluster_stats(clusters)

    sizes = sorted([c.member_count for c in clusters])

    # Size distribution
    size_dist: dict[int, int] = {}
    for s in sizes:
        size_dist[s] = size_dist.get(s, 0) + 1

    # Largest clusters (top 10)
    largest = sorted(clusters, key=lambda c: -c.member_count)[:10]
    largest_dicts = [
        {"cluster_id": c.cluster_id, "size": c.member_count, "representative_id": c.representative_id}
        for c in largest
    ]

    # Smallest clusters (bottom 10)
    smallest = sorted(clusters, key=lambda c: c.member_count)[:10]
    smallest_dicts = [
        {"cluster_id": c.cluster_id, "size": c.member_count, "representative_id": c.representative_id}
        for c in smallest
    ]

    # Quality distribution
    quality_dist = stats.quality_distribution

    # Top representatives by quality
    top_repr = sorted(clusters, key=lambda c: -c.quality_score)[:20]
    top_repr_ids = [c.representative_id for c in top_repr]

    # Singletons (clusters of size 1)
    singleton_count = sum(1 for c in clusters if c.member_count == 1)

    # Low quality count
    low_quality_count = sum(1 for c in clusters if c.cluster_type.value == "low_quality")

    # Overall density
    avg_density = stats.average_density

    return ClusterReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=round(elapsed_seconds, 2),
        total_clusters=stats.total_clusters,
        total_members=stats.total_members,
        total_relationships=relationship_count,
        cluster_size_distribution=size_dist,
        largest_clusters=largest_dicts,
        smallest_clusters=smallest_dicts,
        average_cluster_size=stats.average_cluster_size,
        cluster_density=avg_density,
        quality_distribution=quality_dist,
        top_representative_ids=top_repr_ids,
        orphan_concept_count=orphan_concept_count,
        singleton_count=singleton_count,
        low_quality_count=low_quality_count,
        provider=provider,
        algorithm=algorithm,
    )
