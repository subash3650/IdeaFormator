"""RelationshipStatistics – comprehensive statistics for a relationship set.

Separated from the engine for independent reuse.  Responsible for:
- similarity histogram
- confidence histogram
- degree distribution
- average / max neighbours
- isolated / connected node counts
- relationship type distribution
- source/target distribution
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class RelationshipStatistics:
    """Expanded statistics for a relationship DataFrame."""

    # Core counts
    total_relationships: int = 0
    unique_source_ids: int = 0
    unique_target_ids: int = 0
    unique_pair_ids: int = 0

    # Similarity
    avg_similarity: float = 0.0
    std_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    similarity_histogram_bins: list[float] = field(default_factory=list)
    similarity_histogram_counts: list[int] = field(default_factory=list)

    # Confidence
    avg_confidence: float = 0.0
    std_confidence: float = 0.0
    confidence_histogram_bins: list[float] = field(default_factory=list)
    confidence_histogram_counts: list[int] = field(default_factory=list)

    # Degree distribution
    average_neighbors: float = 0.0
    max_neighbors: int = 0
    min_neighbors: int = 0
    isolated_nodes: int = 0
    connected_nodes: int = 0
    degree_histogram_bins: list[int] = field(default_factory=list)
    degree_histogram_counts: list[int] = field(default_factory=list)

    # Types
    relationship_type_counts: dict[str, int] = field(default_factory=dict)
    source_type_counts: dict[str, int] = field(default_factory=dict)
    target_type_counts: dict[str, int] = field(default_factory=dict)

    # Density
    density: float = 0.0

    # Top connected
    top_connected_sources: list[dict] = field(default_factory=list)
    top_connected_targets: list[dict] = field(default_factory=list)


def compute_relationship_statistics(
    df: pl.DataFrame,
    total_source_items: int = 0,
    similarity_bins: int = 10,
    confidence_bins: int = 10,
    top_k: int = 10,
) -> RelationshipStatistics:
    """Compute comprehensive statistics from a relationship DataFrame.

    Args:
        df: Relationship DataFrame.
        total_source_items: Total items in the source collection (for density).
        similarity_bins: Number of histogram bins for similarity.
        confidence_bins: Number of histogram bins for confidence.
        top_k: Number of top connected nodes to report.

    Returns:
        A fully populated RelationshipStatistics dataclass.
    """
    stats = RelationshipStatistics()
    stats.total_relationships = df.height

    if df.height == 0:
        return stats

    # Source/target IDs
    if "source_id" in df.columns:
        stats.unique_source_ids = df["source_id"].n_unique()
    if "target_id" in df.columns:
        stats.unique_target_ids = df["target_id"].n_unique()

    # Unique pairs
    if "source_id" in df.columns and "target_id" in df.columns:
        pairs = df.select([
            pl.min_horizontal("source_id", "target_id").alias("a"),
            pl.max_horizontal("source_id", "target_id").alias("b"),
        ]).unique()
        stats.unique_pair_ids = pairs.height

    # Similarity stats + histogram
    if "similarity_score" in df.columns:
        sims = df["similarity_score"].to_numpy().astype(np.float64)
        stats.avg_similarity = float(sims.mean())
        stats.std_similarity = float(sims.std())
        stats.min_similarity = float(sims.min())
        stats.max_similarity = float(sims.max())

        bins = np.linspace(float(sims.min()), float(sims.max()), similarity_bins + 1)
        hist, _ = np.histogram(sims, bins=bins)
        stats.similarity_histogram_bins = bins.tolist()
        stats.similarity_histogram_counts = hist.tolist()

    # Confidence stats + histogram
    if "confidence" in df.columns:
        confs = df["confidence"].to_numpy().astype(np.float64)
        stats.avg_confidence = float(confs.mean())
        stats.std_confidence = float(confs.std())

        bins = np.linspace(float(confs.min()), float(confs.max()), confidence_bins + 1)
        hist, _ = np.histogram(confs, bins=bins)
        stats.confidence_histogram_bins = bins.tolist()
        stats.confidence_histogram_counts = hist.tolist()

    # Type counts
    if "source_type" in df.columns:
        stats.source_type_counts = dict(df["source_type"].value_counts().rows())
    if "target_type" in df.columns:
        stats.target_type_counts = dict(df["target_type"].value_counts().rows())
    if "relationship_type" in df.columns:
        stats.relationship_type_counts = dict(df["relationship_type"].value_counts().rows())

    # Density
    if total_source_items > 1:
        possible_pairs = total_source_items * (total_source_items - 1) / 2
        stats.density = stats.unique_pair_ids / possible_pairs if possible_pairs > 0 else 0.0
    elif stats.unique_source_ids > 1:
        # Fall back to computing density from observed source IDs
        possible_pairs = stats.unique_source_ids * (stats.unique_source_ids - 1) / 2
        stats.density = stats.unique_pair_ids / possible_pairs if possible_pairs > 0 else 0.0

    # Degree distribution
    source_degree = Counter(df["source_id"].to_list())
    target_degree = Counter(df["target_id"].to_list())
    all_degrees = list(source_degree.values()) + list(target_degree.values())

    total_nodes = len(
        set(list(source_degree.keys()) + list(target_degree.keys()))
    )
    stats.connected_nodes = len(source_degree)
    # Isolated nodes: total source items minus those that have at least one relationship
    stats.isolated_nodes = max(0, total_source_items - stats.unique_source_ids)
    stats.average_neighbors = (
        round(float(sum(all_degrees) / max(len(all_degrees), 1)), 4)
    )
    stats.max_neighbors = max(all_degrees) if all_degrees else 0
    stats.min_neighbors = min(all_degrees) if all_degrees else 0

    # Degree histogram (log-scale bins for skewed distributions)
    if all_degrees:
        max_d = max(all_degrees)
        if max_d > 1:
            degree_bins = list(range(1, max_d + 2))
            d_hist, _ = np.histogram(all_degrees, bins=degree_bins)
            stats.degree_histogram_bins = degree_bins[:-1]
            stats.degree_histogram_counts = d_hist.tolist()

    # Top connected
    stats.top_connected_sources = [
        {"source_id": sid, "relationship_count": cnt}
        for sid, cnt in source_degree.most_common(top_k)
    ]
    stats.top_connected_targets = [
        {"target_id": tid, "relationship_count": cnt}
        for tid, cnt in target_degree.most_common(top_k)
    ]

    return stats