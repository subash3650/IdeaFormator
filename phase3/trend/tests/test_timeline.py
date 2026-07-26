"""Tests for TrendTimeline."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phase3.trend.schema import TrendSnapshot
from phase3.trend.snapshot import TrendSnapshotBuilder
from phase3.trend.timeline import TrendTimeline, TrendTimelineBuilder, TrendTimelinePoint


class TestTrendTimelinePoint:
    def test_properties(self) -> None:
        snap = TrendSnapshot(
            snapshot_id="s1", run_id="run1", timestamp="2026-01-01T00:00:00"
        )
        point = TrendTimelinePoint(snap, {"obs": 100})
        assert point.timestamp == "2026-01-01T00:00:00"
        assert point.run_id == "run1"
        assert point.metrics["obs"] == 100


class TestTrendTimeline:
    def test_empty(self) -> None:
        tl = TrendTimeline([])
        assert tl.count == 0
        assert tl.first() is None
        assert tl.last() is None
        assert tl.duration_days() == 0.0

    def test_single_point(self) -> None:
        snap = TrendSnapshot(
            snapshot_id="s1", run_id="run1", timestamp="2026-01-01T00:00:00"
        )
        tl = TrendTimeline([TrendTimelinePoint(snap)])
        assert tl.count == 1
        assert tl.first() is not None
        assert tl.last() is not None

    def test_multi_point(self) -> None:
        snaps = [
            TrendSnapshot(snapshot_id="s1", run_id="r1", timestamp="2026-01-01T00:00:00"),
            TrendSnapshot(snapshot_id="s2", run_id="r2", timestamp="2026-01-15T00:00:00"),
            TrendSnapshot(snapshot_id="s3", run_id="r3", timestamp="2026-02-01T00:00:00"),
        ]
        points = [TrendTimelinePoint(s) for s in snaps]
        tl = TrendTimeline(points)
        assert tl.count == 3
        assert tl.first().run_id == "r1"
        assert tl.last().run_id == "r3"

    def test_window(self) -> None:
        snaps = [
            TrendSnapshot(snapshot_id="s1", run_id="r1", timestamp="2026-01-01T00:00:00"),
            TrendSnapshot(snapshot_id="s2", run_id="r2", timestamp="2026-01-15T00:00:00"),
            TrendSnapshot(snapshot_id="s3", run_id="r3", timestamp="2026-02-01T00:00:00"),
        ]
        points = [TrendTimelinePoint(s) for s in snaps]
        tl = TrendTimeline(points)
        w = tl.window(2)
        assert w.count == 2
        assert w.first().run_id == "r2"

    def test_duration_days(self) -> None:
        snaps = [
            TrendSnapshot(snapshot_id="s1", run_id="r1", timestamp="2026-01-01T00:00:00"),
            TrendSnapshot(snapshot_id="s2", run_id="r2", timestamp="2026-01-11T00:00:00"),
        ]
        points = [TrendTimelinePoint(s) for s in snaps]
        tl = TrendTimeline(points)
        assert tl.duration_days() == 10.0

    def test_sort_order(self) -> None:
        snaps = [
            TrendSnapshot(snapshot_id="s2", run_id="r2", timestamp="2026-02-01T00:00:00"),
            TrendSnapshot(snapshot_id="s1", run_id="r1", timestamp="2026-01-01T00:00:00"),
        ]
        points = [TrendTimelinePoint(s) for s in snaps]
        tl = TrendTimeline(points)
        assert tl.first().run_id == "r1"
        assert tl.last().run_id == "r2"
