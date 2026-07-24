"""Exact brute-force linear index using NumPy."""

from __future__ import annotations

import numpy as np

from phase2.similarity.indexes.base import VectorIndex


class LinearIndex(VectorIndex):
    """Exact nearest-neighbour search via brute-force NumPy.

    Suitable for datasets up to ~100k vectors. For larger datasets,
    consider FAISS or HNSW implementations.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got {vectors.ndim}D")
        self._vectors = vectors.astype(np.float32)
        self._size = vectors.shape[0]
        self._dim = vectors.shape[1]

    @property
    def size(self) -> int:
        return self._size

    @property
    def dimension(self) -> int:
        return self._dim

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=np.float32).flatten()
        scores = self._vectors @ query
        top_k = min(k, self._size)
        if top_k == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        indices = np.argpartition(scores, -top_k)[-top_k:]
        order = np.argsort(-scores[indices])
        indices = indices[order]
        return scores[indices], indices

    def search_batch(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        scores = queries @ self._vectors.T  # [n, m]
        n = queries.shape[0]
        top_k = min(k, self._size)
        if top_k == 0:
            return np.zeros((n, 0), dtype=np.float32), np.zeros((n, 0), dtype=np.int64)
        indices = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        # Sort each row
        row_indices = np.arange(n)[:, np.newaxis]
        order = np.argsort(-scores[row_indices, indices], axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        scores_out = np.take_along_axis(scores, indices, axis=1)
        return scores_out, indices
