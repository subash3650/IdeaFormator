"""Export manifest, quality report, and JSON report for relationship generation."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from phase2.similarity.metrics import RelationshipStats
from phase2.similarity.schema import RelationshipManifest
from phase2.similarity.threshold import ThresholdRecommendation


def write_manifest(
    manifest: RelationshipManifest,
    output_dir: Path,
) -> Path:
    """Write the relationship manifest as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "similarity_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2, default=str)
    return path


def write_json_report(
    stats: RelationshipStats,
    output_dir: Path,
    elapsed_seconds: float = 0.0,
    run_id: str | None = None,
    threshold_rec: ThresholdRecommendation | None = None,
    filter_counts: dict[str, int] | None = None,
    df: pl.DataFrame | None = None,
) -> Path:
    """Generate a structured JSON report consumable by downstream modules.

    Includes generation metadata, similarity/confidence statistics,
    threshold recommendations, filter statistics, and degree distribution.
    """
    report: dict = {
        "report_type": "similarity_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id or "N/A",
        "elapsed_seconds": round(elapsed_seconds, 2),
        "relationship_count": stats.total_relationships,
        "unique_source_ids": stats.unique_source_ids,
        "unique_target_ids": stats.unique_target_ids,
        "unique_pairs": stats.unique_pair_ids,
        "similarity": {
            "mean": round(stats.avg_similarity, 6) if stats.total_relationships > 0 else 0.0,
            "std": round(stats.std_similarity, 6) if stats.total_relationships > 0 else 0.0,
            "min": round(stats.min_similarity, 6) if stats.total_relationships > 0 else 0.0,
            "max": round(stats.max_similarity, 6) if stats.total_relationships > 0 else 0.0,
        },
        "confidence": {
            "mean": round(stats.avg_confidence, 6) if stats.total_relationships > 0 else 0.0,
            "std": round(stats.std_confidence, 6) if stats.total_relationships > 0 else 0.0,
        },
        "density": round(stats.density, 8),
        "source_type_counts": dict(stats.source_type_counts),
        "target_type_counts": dict(stats.target_type_counts),
    }

    # Threshold recommendations
    if threshold_rec is not None:
        rec = {
            "min_similarity": round(threshold_rec.min_similarity, 6),
            "max_similarity": round(threshold_rec.max_similarity, 6),
            "mean": round(threshold_rec.mean, 6),
            "median": round(threshold_rec.median, 6),
            "std": round(threshold_rec.std, 6),
            "percentiles": {
                "p50": round(threshold_rec.p50, 6),
                "p75": round(threshold_rec.p75, 6),
                "p90": round(threshold_rec.p90, 6),
                "p95": round(threshold_rec.p95, 6),
                "p99": round(threshold_rec.p99, 6),
            },
            "sample_size": threshold_rec.sample_size,
            "recommended_range": {
                "min": round(threshold_rec.recommended_min, 6),
                "max": round(threshold_rec.recommended_max, 6),
            },
            "candidate_thresholds": {
                str(k): v for k, v in threshold_rec.candidate_thresholds.items()
            },
            "configured_threshold": threshold_rec.configured_threshold,
        }
        if threshold_rec.threshold_warning:
            rec["warning"] = threshold_rec.threshold_warning
        report["threshold_recommendations"] = rec

    # Filter statistics
    if filter_counts:
        report["filter_statistics"] = {
            "stages": dict(filter_counts),
        }

    # Degree distribution from DataFrame
    if df is not None and df.height > 0:
        source_degrees = Counter(df["source_id"].to_list())
        target_degrees = Counter(df["target_id"].to_list())
        all_degrees = list(source_degrees.values()) + list(target_degrees.values())

        report["degree_distribution"] = {
            "total_nodes": len(set(list(source_degrees.keys()) + list(target_degrees.keys()))),
            "average_neighbors": round(float(sum(all_degrees) / max(len(all_degrees), 1)), 4),
            "max_neighbors": max(all_degrees) if all_degrees else 0,
            "min_neighbors": min(all_degrees) if all_degrees else 0,
            "isolated_nodes": stats.unique_source_ids - len(source_degrees) if stats.total_relationships > 0 else 0,
            "connected_nodes": len(source_degrees) if stats.total_relationships > 0 else 0,
        }

        # Top connected concepts
        top_source = source_degrees.most_common(10)
        top_target = target_degrees.most_common(10)
        report["top_connected_sources"] = [
            {"source_id": sid, "relationship_count": cnt} for sid, cnt in top_source
        ]
        report["top_connected_targets"] = [
            {"target_id": tid, "relationship_count": cnt} for tid, cnt in top_target
        ]

        # Similarity histogram (10 bins)
        sims = df["similarity_score"].to_numpy()
        bins = [round(0.7 + i * 0.03, 2) for i in range(11)]
        hist, _ = np.histogram(sims, bins=bins)
        report["similarity_histogram"] = {
            "bins": bins,
            "counts": hist.tolist(),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "similarity_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def generate_quality_report(
    stats: RelationshipStats,
    output_dir: Path,
    elapsed_seconds: float = 0.0,
    run_id: str | None = None,
    threshold_rec: ThresholdRecommendation | None = None,
    filter_counts: dict[str, int] | None = None,
    configured_threshold: float = 0.82,
) -> Path:
    """Generate a human-readable quality report."""
    w = threshold_rec  # shorthand

    lines = [
        "=" * 60,
        "Semantic Relationship Quality Report",
        "=" * 60,
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Run ID:    {run_id or 'N/A'}",
        "",
        "--- Overview ---",
        f"Total relationships : {stats.total_relationships}",
        f"Unique source IDs   : {stats.unique_source_ids}",
        f"Unique target IDs   : {stats.unique_target_ids}",
        f"Unique pairs        : {stats.unique_pair_ids}",
        f"Relationship density: {stats.density:.6f}",
        "",
        "--- Similarity ---",
        f"Mean similarity : {stats.avg_similarity:.6f}",
        f"Std similarity  : {stats.std_similarity:.6f}",
        f"Min similarity  : {stats.min_similarity:.6f}",
        f"Max similarity  : {stats.max_similarity:.6f}",
        "",
        "--- Confidence ---",
        f"Mean confidence : {stats.avg_confidence:.6f}",
        f"Std confidence  : {stats.std_confidence:.6f}",
        "",
        "--- By Source Type ---",
    ]
    for src, count in sorted(stats.source_type_counts.items()):
        lines.append(f"  {src}: {count}")
    lines.extend([
        "",
        "--- By Target Type ---",
    ])
    for tgt, count in sorted(stats.target_type_counts.items()):
        lines.append(f"  {tgt}: {count}")
    lines.extend([
        "",
        f"Elapsed time: {elapsed_seconds:.2f}s",
        "",
        "Checks",
        "------",
        f"  Relationships present : {'PASS' if stats.total_relationships > 0 else 'FAIL'}",
        f"  IDs unique per pair   : {'PASS' if stats.unique_pair_ids <= stats.total_relationships else 'WARN'}",
        f"  Similarity in [0,1]   : {'PASS' if 0.0 <= stats.min_similarity and stats.max_similarity <= 1.0 else 'WARN'}",
        f"  Confidence in [0,1]   : {'PASS' if stats.avg_confidence >= 0.0 else 'WARN'}",
    ])

    # Threshold recommendations
    if w is not None and w.sample_size > 0:
        lines.extend([
            "",
            "--- Threshold Recommendations ---",
            f"  Configuration threshold : {configured_threshold:.2f}",
            f"  Score distribution      : {w.sample_size} samples",
            f"  Mean      : {w.mean:.4f}",
            f"  Median    : {w.median:.4f}",
            f"  Std       : {w.std:.4f}",
            f"  P50       : {w.p50:.4f}",
            f"  P75       : {w.p75:.4f}",
            f"  P90       : {w.p90:.4f}",
            f"  P95       : {w.p95:.4f}",
            f"  P99       : {w.p99:.4f}",
            f"  Recommended range: [{w.recommended_min:.2f}, {w.recommended_max:.2f}]",
            "",
            "  Estimated Relationships by Threshold:",
        ])
        for t, cnt in sorted(w.candidate_thresholds.items()):
            lines.append(f"    {t:.2f}: {cnt}")
        if w.threshold_warning:
            lines.extend([
                "",
                f"  [!] WARNING: {w.threshold_warning}",
            ])

    # Filter statistics
    if filter_counts:
        lines.extend([
            "",
            "--- Filter Pipeline ---",
        ])
        for stage, count in filter_counts.items():
            lines.append(f"  {stage}: {count}")

    lines.append("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "similarity_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path