"""TrendCache — reuse trend results when upstream data hasn't changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phase3.trend.config import TrendConfig
from phase3.trend.schema import TrendOutput
from phase3.trend.store import TrendStore


class TrendCache:
    """Cache that invalidates when upstream reasoning, KG, or opportunity checksums change."""

    def __init__(self, store: TrendStore, config: TrendConfig) -> None:
        self._store = store
        self._config = config

    def _cache_marker_path(self) -> Path:
        return self._store.trend_dir / ".trend_cache"

    def _compute_input_hash(
        self,
        reasoning_checksums: dict | None,
        kg_checksums: dict | None,
        opportunity_checksums: dict | None,
    ) -> str:
        h = hashlib.sha256()
        h.update(self._config.model_dump_json().encode())
        if reasoning_checksums:
            h.update(json.dumps(reasoning_checksums, sort_keys=True).encode())
        if kg_checksums:
            h.update(json.dumps(kg_checksums, sort_keys=True).encode())
        if opportunity_checksums:
            h.update(json.dumps(opportunity_checksums, sort_keys=True).encode())
        return h.hexdigest()[:32]

    def is_valid(
        self,
        reasoning_checksums: dict | None,
        kg_checksums: dict | None,
        opportunity_checksums: dict | None,
    ) -> bool:
        if not self._config.cache_enabled:
            return False
        marker = self._cache_marker_path()
        if not marker.exists():
            return False
        expected = self._compute_input_hash(reasoning_checksums, kg_checksums, opportunity_checksums)
        actual = marker.read_text(encoding="utf-8").strip()
        return actual == expected

    def save(
        self,
        reasoning_checksums: dict | None,
        kg_checksums: dict | None,
        opportunity_checksums: dict | None,
    ) -> None:
        h = self._compute_input_hash(reasoning_checksums, kg_checksums, opportunity_checksums)
        self._cache_marker_path().write_text(h, encoding="utf-8")

    def load(self) -> TrendOutput | None:
        trends = self._store.load_trends()
        metadata = self._store.load_metadata()
        if not trends:
            return None
        return TrendOutput(trends=trends, metadata=metadata)

    def invalidate(self) -> None:
        marker = self._cache_marker_path()
        if marker.exists():
            marker.unlink()
