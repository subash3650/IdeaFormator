"""Tests for OpportunityCache."""

from __future__ import annotations

from pathlib import Path

from phase3.opportunity.cache import OpportunityCache
from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.schema import Opportunity
from phase3.opportunity.store import OpportunityStore


class TestOpportunityCache:
    def test_invalid_when_no_marker(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        store = OpportunityStore(tmp_path)
        cache = OpportunityCache(store, cfg)
        assert cache.is_valid(None, None) is False

    def test_invalid_when_checksum_mismatch(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        store = OpportunityStore(tmp_path)
        cache = OpportunityCache(store, cfg)
        cache.save({"reasoning": "abc"}, {"kg": "def"})
        assert cache.is_valid({"reasoning": "xyz"}, {"kg": "def"}) is False

    def test_valid_when_checksums_match(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        store = OpportunityStore(tmp_path)
        cache = OpportunityCache(store, cfg)
        cache.save({"reasoning": "abc"}, {"kg": "def"})
        assert cache.is_valid({"reasoning": "abc"}, {"kg": "def"}) is True

    def test_save_and_load(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        store = OpportunityStore(tmp_path)
        cache = OpportunityCache(store, cfg)
        store.save_opportunities([
            Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p"),
        ], "run1")
        store.save_metadata(
            __import__("phase3.opportunity.schema", fromlist=["OpportunityMetadata"])
            .OpportunityMetadata(run_id="run1")
        )
        cache.save({"reasoning": "abc"}, {"kg": "def"})
        loaded = cache.load()
        assert loaded is not None
        assert len(loaded.opportunities) == 1

    def test_cache_disabled(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, cache_enabled=False)
        store = OpportunityStore(tmp_path)
        cache = OpportunityCache(store, cfg)
        cache.save({"r": "a"}, {"k": "b"})
        assert cache.is_valid({"r": "a"}, {"k": "b"}) is False

    def test_invalidate(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        store = OpportunityStore(tmp_path)
        cache = OpportunityCache(store, cfg)
        cache.save({"r": "a"}, {"k": "b"})
        assert cache.is_valid({"r": "a"}, {"k": "b"}) is True
        cache.invalidate()
        assert cache.is_valid({"r": "a"}, {"k": "b"}) is False
