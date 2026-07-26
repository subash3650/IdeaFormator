"""Tests for TrendCache."""

from __future__ import annotations

from pathlib import Path

from phase3.trend.cache import TrendCache
from phase3.trend.config import TrendConfig
from phase3.trend.schema import (
    Trend,
    TrendDirection,
    TrendMetadata,
    TrendSubject,
    TrendType,
)
from phase3.trend.store import TrendStore


class TestTrendCache:
    def test_invalid_when_no_marker(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        store = TrendStore(tmp_path)
        cache = TrendCache(store, cfg)
        assert cache.is_valid(None, None, None) is False

    def test_invalid_when_checksum_mismatch(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        store = TrendStore(tmp_path)
        cache = TrendCache(store, cfg)
        cache.save({"reasoning": "abc"}, {"kg": "def"}, {"opp": "ghi"})
        assert cache.is_valid({"reasoning": "xyz"}, {"kg": "def"}, {"opp": "ghi"}) is False

    def test_valid_when_checksums_match(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        store = TrendStore(tmp_path)
        cache = TrendCache(store, cfg)
        cache.save({"reasoning": "abc"}, {"kg": "def"}, {"opp": "ghi"})
        assert cache.is_valid({"reasoning": "abc"}, {"kg": "def"}, {"opp": "ghi"}) is True

    def test_save_and_load(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        store = TrendStore(tmp_path)
        cache = TrendCache(store, cfg)
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
        )
        store.save_trends([t], "run1")
        store.save_metadata(TrendMetadata(run_id="run1"))
        cache.save({"r": "a"}, {"k": "b"}, {"o": "c"})
        loaded = cache.load()
        assert loaded is not None
        assert len(loaded.trends) == 1

    def test_cache_disabled(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path, cache_enabled=False)
        store = TrendStore(tmp_path)
        cache = TrendCache(store, cfg)
        cache.save({"r": "a"}, {"k": "b"}, {"o": "c"})
        assert cache.is_valid({"r": "a"}, {"k": "b"}, {"o": "c"}) is False

    def test_invalidate(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path)
        store = TrendStore(tmp_path)
        cache = TrendCache(store, cfg)
        cache.save({"r": "a"}, {"k": "b"}, {"o": "c"})
        assert cache.is_valid({"r": "a"}, {"k": "b"}, {"o": "c"}) is True
        cache.invalidate()
        assert cache.is_valid({"r": "a"}, {"k": "b"}, {"o": "c"}) is False
