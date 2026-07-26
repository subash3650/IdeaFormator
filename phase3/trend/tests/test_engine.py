"""Tests for TrendEngine."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phase3.trend.config import TrendConfig
from phase3.trend.schema import Trend, TrendDirection, TrendSubject, TrendType
from phase3.trend.snapshot import TrendSnapshotBuilder


class TestTrendEngine:
    def test_stats_empty(self, tmp_path: Path) -> None:
        from phase3.trend.engine import TrendEngine
        cfg = TrendConfig(output_dir=tmp_path)
        engine = TrendEngine(cfg)
        stats = engine.stats()
        assert stats["total_trends"] == 0

    def test_stats_after_add(self, tmp_path: Path) -> None:
        from phase3.trend.engine import TrendEngine
        cfg = TrendConfig(output_dir=tmp_path)
        engine = TrendEngine(cfg)
        engine.store.save_trends([
            Trend(
                trend_id="t1", title="T", summary="S",
                trend_type=TrendType.GROWING,
                trend_direction=TrendDirection.UP,
                trend_subject=TrendSubject.PROBLEM,
                subject_id="p1", subject_label="P1",
            ),
        ], "run1")
        stats = engine.stats()
        assert stats["total_trends"] == 1
        assert stats["growing"] == 1

    def test_search(self, tmp_path: Path) -> None:
        from phase3.trend.engine import TrendEngine
        cfg = TrendConfig(output_dir=tmp_path)
        engine = TrendEngine(cfg)
        engine.store.save_trends([
            Trend(
                trend_id="t1", title="Growing Problem",
                summary="Users report increasing issues",
                trend_type=TrendType.GROWING,
                trend_direction=TrendDirection.UP,
                trend_subject=TrendSubject.PROBLEM,
                subject_id="p1", subject_label="Problem1",
            ),
        ], "run1")
        results = engine.search("Growing")
        assert len(results) > 0

    def test_search_no_match(self, tmp_path: Path) -> None:
        from phase3.trend.engine import TrendEngine
        cfg = TrendConfig(output_dir=tmp_path)
        engine = TrendEngine(cfg)
        results = engine.search("nonexistent")
        assert results == []

    def test_create_snapshot(self, tmp_path: Path) -> None:
        from phase3.trend.engine import TrendEngine
        cfg = TrendConfig(output_dir=tmp_path)
        engine = TrendEngine(cfg, run_id="manual_test")
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"id": ["1"]}).write_parquet(str(assets_dir / "observations.parquet"))
        result = engine.create_snapshot(tmp_path)
        assert result["run_id"] == "manual_test"
        assert result["snapshot_id"] != ""

    def test_generate_without_snapshots(self, tmp_path: Path) -> None:
        from phase3.trend.engine import TrendEngine
        cfg = TrendConfig(output_dir=tmp_path)
        engine = TrendEngine(cfg)
        result = engine.generate(tmp_path)
        assert result["total_trends"] == 0

    def test_clear_cache(self, tmp_path: Path) -> None:
        from phase3.trend.engine import TrendEngine
        cfg = TrendConfig(output_dir=tmp_path)
        engine = TrendEngine(cfg)
        engine.clear_cache()
        assert True
