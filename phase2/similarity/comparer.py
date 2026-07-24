"""High-performance NumPy similarity computation helpers."""

from __future__ import annotations

import numpy as np

from phase2.similarity.providers.base import SimilarityProvider


def top_k_similarity(
    query_vectors: np.ndarray,
    index_vectors: np.ndarray,
    provider: SimilarityProvider,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-k similarity for a batch of queries against an index.

    Args:
        query_vectors: Shape [n, d]
        index_vectors: Shape [m, d]
        provider: Similarity provider
        k: Number of neighbours

    Returns:
        (scores, indices) each of shape [n, k]
    """
    n = query_vectors.shape[0]
    m = index_vectors.shape[0]
    top_k = min(k, m)

    if top_k == 0:
        return np.zeros((n, 0), dtype=np.float32), np.zeros((n, 0), dtype=np.int64)

    # Compute all pairwise scores: [n, m]
    all_scores = provider.compute_pairwise(query_vectors, index_vectors)

    # For each query, find top-k
    scores_out = np.zeros((n, top_k), dtype=np.float32)
    indices_out = np.zeros((n, top_k), dtype=np.int64)

    for i in range(n):
        row_scores = all_scores[i]
        top_indices = np.argpartition(row_scores, -top_k)[-top_k:]
        order = np.argsort(-row_scores[top_indices])
        top_indices = top_indices[order]
        scores_out[i] = row_scores[top_indices]
        indices_out[i] = top_indices

    return scores_out, indices_out


def count_frequencies(source_ids: list[str]) -> dict[str, int]:
    """Count occurrences of each source ID."""
    freq: dict[str, int] = {}
    for sid in source_ids:
        freq[sid] = freq.get(sid, 0) + 1
    return freq
