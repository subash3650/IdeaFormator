"""Tests for Trend Snapshot system."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phase3.trend.snapshot import (
    TrendSnapshotBuilder,
    TrendSnapshotComparator,
    TrendSnapshotLoader,
    TrendSnapshotScanner,
)


class TestTrendSnapshotScanner:
    def test_scan_empty(self, tmp_path: Path) -> None:
        scanner = TrendSnapshotScanner(tmp_path)
        snaps = scanner.scan()
        assert snaps == []

    def test_count_zero(self, tmp_path: Path) -> None:
        scanner = TrendSnapshotScanner(tmp_path)
        assert scanner.count() == 0

    def test_latest_empty(self, tmp_path: Path) -> None:
        scanner = TrendSnapshotScanner(tmp_path)
        assert scanner.latest() is None

    def test_prior_empty(self, tmp_path: Path) -> None:
        scanner = TrendSnapshotScanner(tmp_path)
        assert scanner.prior() is None

    def test_snapshots_dir(self, tmp_path: Path) -> None:
        scanner = TrendSnapshotScanner(tmp_path)
        assert scanner.snapshots_dir == tmp_path / "snapshots"

    def test_scan_with_created_snapshot(self, tmp_path: Path) -> None:
        builder = TrendSnapshotBuilder(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        # Create a placeholder observations file
        df = pl.DataFrame({"id": ["1"]})
        df.write_parquet(str(assets_dir / "observations.parquet"))
        builder.create("run1")
        scanner = TrendSnapshotScanner(tmp_path)
        snaps = scanner.scan()
        assert len(snaps) == 1
        assert snaps[0].run_id == "run1"


class TestTrendSnapshotBuilder:
    def test_create_snapshot(self, tmp_path: Path) -> None:
        builder = TrendSnapshotBuilder(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame({"id": ["1", "2"], "value": [1.0, 2.0]})
        df.write_parquet(str(assets_dir / "observations.parquet"))
        for fname in builder.ASSET_FILES[1:]:
            df.write_parquet(str(assets_dir / fname))

        snapshot = builder.create("run1")
        assert snapshot.run_id == "run1"
        assert snapshot.snapshot_id != ""
        assert len(snapshot.asset_checksums) > 0
        assert (tmp_path / "snapshots" / "run1" / "snapshot_manifest.json").exists()
        assert (tmp_path / "snapshots" / ".snapshot_index.json").exists()

    def test_create_empty_snapshot(self, tmp_path: Path) -> None:
        builder = TrendSnapshotBuilder(tmp_path)
        snapshot = builder.create("run_empty")
        assert snapshot.run_id == "run_empty"
        assert snapshot.observation_count == 0


class TestTrendSnapshotLoader:
    def test_load_nonexistent_asset(self, tmp_path: Path) -> None:
        from phase3.trend.schema import TrendSnapshot
        snap = TrendSnapshot(
            snapshot_id="s1", run_id="run1", timestamp="2026-01-01T00:00:00"
        )
        loader = TrendSnapshotLoader(tmp_path)
        df = loader.load_asset(snap, "nonexistent")
        assert df.height == 0

    def test_load_asset(self, tmp_path: Path) -> None:
        builder = TrendSnapshotBuilder(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame({"id": ["1"], "val": [1]})
        df.write_parquet(str(assets_dir / "observations.parquet"))
        snap = builder.create("run1")

        loader = TrendSnapshotLoader(tmp_path)
        loaded = loader.load_asset(snap, "observations")
        assert loaded.height == 1

    def test_load_observations(self, tmp_path: Path) -> None:
        builder = TrendSnapshotBuilder(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"id": ["1"]}).write_parquet(str(assets_dir / "observations.parquet"))
        snap = builder.create("run1")
        loader = TrendSnapshotLoader(tmp_path)
        obs = loader.load_observations(snap)
        assert obs.height == 1


class TestTrendSnapshotComparator:
    def test_compare_two_snapshots(self, tmp_path: Path) -> None:
        builder = TrendSnapshotBuilder(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"id": ["1"]}).write_parquet(str(assets_dir / "observations.parquet"))
        snap1 = builder.create("run1")

        pl.DataFrame({"id": ["1", "2"]}).write_parquet(str(assets_dir / "observations.parquet"))
        snap2 = builder.create("run2")

        comparator = TrendSnapshotComparator(tmp_path)
        delta = comparator.compare(snap1, snap2)
        # After/Before counts reflect total rows in each snapshot's observation asset
        assert delta.observation_count_after >= delta.observation_count_before

    def test_compare_identical(self, tmp_path: Path) -> None:
        builder = TrendSnapshotBuilder(tmp_path)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"id": ["1"]}).write_parquet(str(assets_dir / "observations.parquet"))
        snap1 = builder.create("run_a")
        snap2 = builder.create("run_b")

        comparator = TrendSnapshotComparator(tmp_path)
        delta = comparator.compare(snap1, snap2)
        assert delta.observation_growth_pct == 0.0
        assert delta.evidence_growth_pct == 0.0
