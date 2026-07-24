"""Abstract interface for vector index structures.

Responsibility
--------------
Defines the contract for nearest-neighbour retrieval across all index
implementations.  Every index must expose two search methods (single
query and batch) plus read-only properties for size and dimension.

Architecture
------------
    VectorIndex  (abstract — this module)
        |
        +-- LinearIndex   (exact brute-force, small to medium datasets)
        |
        +-- FAISSIndex    *(future)*  Approximate, GPU-accelerated
        |
        +-- QdrantIndex   *(future)*  Distributed vector database
        |
        +-- HNSWIndex     *(future)*  Hierarchical navigable small-world

Extension Points
----------------
To add a new index implementation:

1. Create a new module (e.g. ``faiss_index.py``) in ``indexes/``.
2. Subclass ``VectorIndex`` and implement all abstract methods.
3. Register the class in ``indexes/__init__.py`` for public export.

The engine selects an index class at construction time based on
configuration or dataset size.  The abstract interface keeps the
engine decoupled from any specific index technology.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VectorIndex(ABC):
    """Abstract vector index for nearest-neighbour retrieval.

    Every concrete implementation must support:

    * ``search(query, k)`` — single-query top-k search
    * ``search_batch(queries, k)`` — batched top-k search
    * ``size`` — number of indexed vectors
    * ``dimension`` — vector dimensionality

    Implementation notes for future index types
    --------------------------------------------
    **FAISSIndex** *(future)*
        - Wrap ``faiss.IndexFlatIP`` (inner product) for GPU or CPU.
        - On construction, call ``faiss.normalize_L2`` on vectors if using
          cosine similarity with ``IndexFlatIP``.
        - ``search_batch`` delegates to ``faiss.Index.search(k)`` which
          natively returns (scores, indices).

    **QdrantIndex** *(future)*
        - Connect to a local or remote Qdrant collection.
        - ``search`` maps to ``client.search(collection, query_vector, limit=k)``.
        - ``search_batch`` maps to ``client.search_batch``.

    **HNSWIndex** *(future)*
        - Use hnswlib or faiss for HNSW graph construction.
        - Set ``ef_construction`` and ``M`` parameters on init.
        - Provide methods to save/load the index to disk.
    """

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Find the k nearest neighbours to a single query vector.

        Args:
            query: Shape ``[d]`` — a single query vector.
            k: Number of nearest neighbours to return.

        Returns:
            A tuple ``(scores, indices)`` each of shape ``[k']`` where
            ``k' = min(k, size)``.  Scores are ordered descending.
        """

    @abstractmethod
    def search_batch(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbours for a batch of query vectors.

        Args:
            queries: Shape ``[n, d]`` — ``n`` query vectors.
            k: Number of nearest neighbours per query.

        Returns:
            A tuple ``(scores, indices)`` each of shape ``[n, k']`` where
            ``k' = min(k, size)``.  Rows are ordered by descending score.
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of vectors currently in the index."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of indexed vectors."""

    def save(self, path: str) -> None:
        """Persist the index to disk.  Override in concrete implementations.

        The default raises ``NotImplementedError``; indexes that support
        serialisation should override this method.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    @classmethod
    def load(cls, path: str) -> VectorIndex:
        """Load a previously saved index from disk.  Override in concrete implementations.

        The default raises ``NotImplementedError``; indexes that support
        deserialisation should override this method.
        """
        raise NotImplementedError(f"{cls.__name__} does not support load()")