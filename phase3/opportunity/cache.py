"""OpportunityCache — reuse opportunities when upstream data hasn't changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.schema import OpportunityOutput
from phase3.opportunity.store import OpportunityStore


class OpportunityCache:
    """Cache that invalidates when upstream reasoning or KG graph checksums change."""

    def __init__(self, store: OpportunityStore, config: OpportunityConfig) -> None:
        self._store = store
        self._config = config

    def _cache_marker_path(self) -> Path:
        return self._store.opportunity_dir / ".opportunity_cache"

    def _compute_input_hash(self, reasoning_checksums: dict | None, kg_checksums: dict | None) -> str:
        h = hashlib.sha256()
        h.update(self._config.model_dump_json().encode())
        if reasoning_checksums:
            h.update(json.dumps(reasoning_checksums, sort_keys=True).encode())
        if kg_checksums:
            h.update(json.dumps(kg_checksums, sort_keys=True).encode())
        return h.hexdigest()[:32]

    def is_valid(self, reasoning_checksums: dict | None, kg_checksums: dict | None) -> bool:
        if not self._config.cache_enabled:
            return False
        marker = self._cache_marker_path()
        if not marker.exists():
            return False
        expected = self._compute_input_hash(reasoning_checksums, kg_checksums)
        actual = marker.read_text(encoding="utf-8").strip()
        return actual == expected

    def save(self, reasoning_checksums: dict | None, kg_checksums: dict | None) -> None:
        h = self._compute_input_hash(reasoning_checksums, kg_checksums)
        self._cache_marker_path().write_text(h, encoding="utf-8")

    def load(self) -> OpportunityOutput | None:
        opportunities = self._store.load_opportunities()
        metadata = self._store.load_metadata()
        if not opportunities:
            return None
        return OpportunityOutput(opportunities=opportunities, metadata=metadata)

    def invalidate(self) -> None:
        marker = self._cache_marker_path()
        if marker.exists():
            marker.unlink()
