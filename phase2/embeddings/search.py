"""Similarity search over embedding vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from phase2.embeddings.schema import SearchResult, SourceType


class VectorIndex(ABC):
    """Abstract interface for nearest-neighbour search."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        """Return the *k* nearest neighbours to *query*."""


class LinearIndex(VectorIndex):
    """Exact cosine-similarity search via brute-force NumPy."""

    def __init__(self, embeddings: np.ndarray, metadata: pl.DataFrame) -> None:
        if embeddings.shape[0] != metadata.height:
            msg = f"Embeddings ({embeddings.shape[0]}) and metadata ({metadata.height}) count mismatch"
            raise ValueError(msg)
        self._embeddings = embeddings
        self._metadata = metadata

    @property
    def size(self) -> int:
        return self._embeddings.shape[0]

    @property
    def dimension(self) -> int:
        return self._embeddings.shape[1]

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        query = np.asarray(query, dtype=np.float32).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        sims = self._embeddings @ query
        top_k = min(k, len(sims))
        indices = np.argpartition(sims, -top_k)[-top_k:]
        indices = indices[np.argsort(-sims[indices])]

        results: list[SearchResult] = []
        for idx in indices:
            row = self._metadata.row(idx, named=True)
            results.append(
                SearchResult(
                    embedding_id=row["embedding_id"],
                    source_id=row["source_id"],
                    source_type=SourceType(row["source_type"]),
                    provider=row["provider"],
                    model=row["model"],
                    similarity=float(sims[idx]),
                    text_snippet=row.get("text_snippet"),
                )
            )
        return results


def build_index(df: pl.DataFrame) -> LinearIndex:
    """Build a LinearIndex from an embedding DataFrame."""
    vecs = np.stack(df["embedding"].to_list()).astype(np.float32)
    meta = df.select([c for c in df.columns if c != "embedding"])
    return LinearIndex(vecs, meta)