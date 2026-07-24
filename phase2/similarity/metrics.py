"""Metrics and statistics for relationship sets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class RelationshipStats:
    """Aggregate statistics for a relationship set."""

    total_relationships: int = 0
    source_type_counts: dict[str, int] = field(default_factory=dict)
    target_type_counts: dict[str, int] = field(default_factory=dict)
    unique_source_ids: int = 0
    unique_target_ids: int = 0
    unique_pair_ids: int = 0
    avg_similarity: float = 0.0
    std_similarity: float = 0.0
    avg_confidence: float = 0.0
    std_confidence: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    density: float = 0.0


def compute_stats(
    df: pl.DataFrame,
    total_source_items: int = 0,
) -> RelationshipStats:
    """Compute aggregate statistics over a relationship DataFrame."""
    stats = RelationshipStats()
    stats.total_relationships = df.height

    if df.height == 0:
        return stats

    if "source_type" in df.columns:
        stats.source_type_counts = dict(df["source_type"].value_counts().rows())
    if "target_type" in df.columns:
        stats.target_type_counts = dict(df["target_type"].value_counts().rows())

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

    if "similarity_score" in df.columns:
        sims = df["similarity_score"].to_numpy().astype(np.float32)
        stats.avg_similarity = float(sims.mean())
        stats.std_similarity = float(sims.std())
        stats.min_similarity = float(sims.min())
        stats.max_similarity = float(sims.max())

    if "confidence" in df.columns:
        confs = df["confidence"].to_numpy().astype(np.float32)
        stats.avg_confidence = float(confs.mean())
        stats.std_confidence = float(confs.std())

    # Density: actual pairs / possible pairs
    if total_source_items > 1:
        possible_pairs = total_source_items * (total_source_items - 1) / 2
        stats.density = stats.unique_pair_ids / possible_pairs if possible_pairs > 0 else 0.0

    return stats
