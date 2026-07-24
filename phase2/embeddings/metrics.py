"""Metrics and statistics for embedding sets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class EmbeddingStats:
    total_vectors: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    dimension: int = 0
    mean_norm: float = 0.0
    std_norm: float = 0.0
    null_text_snippets: int = 0


def compute_stats(df: pl.DataFrame) -> EmbeddingStats:
    """Compute aggregate statistics over an embedding DataFrame."""
    stats = EmbeddingStats()
    stats.total_vectors = df.height

    if "source_type" in df.columns:
        stats.by_source = dict(df["source_type"].value_counts().rows())

    if df.height > 0 and "embedding" in df.columns:
        vecs = np.stack(df["embedding"].to_list()).astype(np.float32)
        stats.dimension = vecs.shape[1]
        norms = np.linalg.norm(vecs, axis=1)
        stats.mean_norm = float(norms.mean())
        stats.std_norm = float(norms.std())

    if "text_snippet" in df.columns:
        stats.null_text_snippets = df["text_snippet"].null_count()

    return stats