"""Lightweight in-memory ID-only cache for embedding deduplication."""

from __future__ import annotations

from collections import OrderedDict


class EmbeddingCache:
    """Stores computed embedding IDs to avoid re-computation.

    Uses an LRU eviction policy when *maxsize* is set (> 0).
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, None] = OrderedDict()

    def contains(self, embedding_id: str) -> bool:
        return embedding_id in self._cache

    def add(self, embedding_id: str) -> None:
        if self._maxsize > 0 and len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[embedding_id] = None

    def remove(self, embedding_id: str) -> None:
        self._cache.pop(embedding_id, None)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def ids(self) -> set[str]:
        return set(self._cache.keys())