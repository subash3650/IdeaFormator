from __future__ import annotations

import time
from typing import Any


class CacheEntry:
    def __init__(self, value: Any, ttl: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl

    @property
    def expired(self) -> bool:
        return time.monotonic() > self.expires_at


class MemoryCache:
    def __init__(self, max_size: int = 5000, default_ttl: float = 60.0) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expired:
            del self._store[key]
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        if len(self._store) >= self._max_size:
            self._evict()
        self._store[key] = CacheEntry(value, ttl or self._default_ttl)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def _evict(self) -> None:
        for k in list(self._store.keys())[: self._max_size // 10]:
            del self._store[k]

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "default_ttl": self._default_ttl,
        }


_cache_instance: MemoryCache | None = None


def get_cache() -> MemoryCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MemoryCache()
    return _cache_instance
