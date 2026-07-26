"""Tests for TrendBuilder."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phase3.trend.builder import TrendBuilder
from phase3.trend.config import TrendConfig
from phase3.trend.snapshot import TrendSnapshotBuilder


class TestTrendBuilder:
    def test_build_without_snapshots(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path, min_snapshots=2)
        builder = TrendBuilder(cfg)
        result = builder.build(tmp_path)
        assert result["total_trends"] == 0

    def test_build_with_two_snapshots(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path, min_snapshots=2, min_growth_pct=0.0)
        builder = TrendBuilder(cfg)

        # Create two snapshots
        snap_builder = TrendSnapshotBuilder(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        pl.DataFrame({"id": ["1", "2"]}).write_parquet(str(assets_dir / "observations.parquet"))
        pl.DataFrame({"id": ["1"]}).write_parquet(str(assets_dir / "evidence.parquet"))
        pl.DataFrame({"id": ["1"]}).write_parquet(str(assets_dir / "problem_signals.parquet"))
        snap_builder.create("run1")

        pl.DataFrame({"id": ["1", "2", "3"]}).write_parquet(str(assets_dir / "observations.parquet"))
        pl.DataFrame({"id": ["1", "2"]}).write_parquet(str(assets_dir / "evidence.parquet"))
        pl.DataFrame({"id": ["1", "2"]}).write_parquet(str(assets_dir / "problem_signals.parquet"))
        snap_builder.create("run2")

        result = builder.build(tmp_path)
        assert result["total_trends"] > 0
        assert result["growing"] > 0 or result["stable"] > 0

    def test_build_saves_to_store(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path, min_snapshots=2, min_growth_pct=0.0)
        builder = TrendBuilder(cfg)
        snap_builder = TrendSnapshotBuilder(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        pl.DataFrame({"id": ["1"]}).write_parquet(str(assets_dir / "observations.parquet"))
        snap_builder.create("r1")
        pl.DataFrame({"id": ["1", "2"]}).write_parquet(str(assets_dir / "observations.parquet"))
        snap_builder.create("r2")

        builder.build(tmp_path)
        loaded = builder.store.load_trends()
        assert len(loaded) > 0

    def test_cache_hit(self, tmp_path: Path) -> None:
        cfg = TrendConfig(output_dir=tmp_path, min_snapshots=2, min_growth_pct=0.0)
        builder = TrendBuilder(cfg)
        snap_builder = TrendSnapshotBuilder(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        pl.DataFrame({"id": ["1"]}).write_parquet(str(assets_dir / "observations.parquet"))
        snap_builder.create("r1")
        pl.DataFrame({"id": ["1", "2"]}).write_parquet(str(assets_dir / "observations.parquet"))
        snap_builder.create("r2")

        result1 = builder.build(tmp_path)
        result2 = builder.build(tmp_path)
        assert result2["cache_hit"] is True
