"""Tests for TrendValidator."""

from __future__ import annotations

from phase3.trend.schema import (
    Trend,
    TrendDirection,
    TrendMetrics,
    TrendSubject,
    TrendType,
)
from phase3.trend.validator import TrendValidator


class TestTrendValidator:
    def test_valid_empty(self) -> None:
        v = TrendValidator()
        result = v.validate([])
        assert result.valid is True
        assert "No trends to validate" in result.warnings

    def test_valid_trends(self) -> None:
        v = TrendValidator()
        trends = [
            Trend(
                trend_id="t1", title="T1", summary="S1",
                trend_type=TrendType.GROWING,
                trend_direction=TrendDirection.UP,
                trend_subject=TrendSubject.PROBLEM,
                subject_id="p1", subject_label="P1",
                metrics=TrendMetrics(growth_pct=50.0),
            ),
            Trend(
                trend_id="t2", title="T2", summary="S2",
                trend_type=TrendType.DECLINING,
                trend_direction=TrendDirection.DOWN,
                trend_subject=TrendSubject.OPPORTUNITY,
                subject_id="o1", subject_label="O1",
                metrics=TrendMetrics(growth_pct=-30.0),
            ),
        ]
        result = v.validate(trends)
        assert result.valid is True
        assert result.trends_checked == 2

    def test_duplicate_ids(self) -> None:
        v = TrendValidator()
        trends = [
            Trend(
                trend_id="t1", title="T", summary="S",
                trend_type=TrendType.GROWING,
                trend_direction=TrendDirection.UP,
                trend_subject=TrendSubject.PROBLEM,
                subject_id="p1", subject_label="P1",
            ),
            Trend(
                trend_id="t1", title="T2", summary="S2",
                trend_type=TrendType.DECLINING,
                trend_direction=TrendDirection.DOWN,
                trend_subject=TrendSubject.PROBLEM,
                subject_id="p2", subject_label="P2",
            ),
        ]
        result = v.validate(trends)
        assert result.valid is False
        assert result.duplicate_count > 0
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_missing_snapshot_reference(self) -> None:
        v = TrendValidator()
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
            snapshot_ids=["nonexistent_snap"],
        )
        result = v.validate([t], valid_snapshot_ids={"real_snap_1"})
        assert result.valid is False
        assert result.missing_snapshot_count > 0

    def test_excess_growth(self) -> None:
        v = TrendValidator()
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
            metrics=TrendMetrics(growth_pct=100000.0),
        )
        result = v.validate([t])
        assert result.valid is False
        assert result.excess_growth_count > 0

    def test_inconsistent_direction_up_with_negative_growth(self) -> None:
        v = TrendValidator()
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.DECLINING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
            metrics=TrendMetrics(growth_pct=-50.0),
        )
        result = v.validate([t])
        assert result.valid is False
        assert result.inconsistent_direction_count > 0

    def test_inconsistent_direction_down_with_positive_growth(self) -> None:
        v = TrendValidator()
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.DOWN,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
            metrics=TrendMetrics(growth_pct=50.0),
        )
        result = v.validate([t])
        assert result.valid is False
        assert result.inconsistent_direction_count > 0

    def test_all_valid_with_snapshots(self) -> None:
        v = TrendValidator()
        trends = [
            Trend(
                trend_id="t1", title="T1", summary="S1",
                trend_type=TrendType.GROWING,
                trend_direction=TrendDirection.UP,
                trend_subject=TrendSubject.PROBLEM,
                subject_id="p1", subject_label="P1",
                snapshot_ids=["s1"],
                metrics=TrendMetrics(growth_pct=50.0),
            ),
        ]
        result = v.validate(trends, valid_snapshot_ids={"s1"})
        assert result.valid is True
        assert result.trends_checked == 1
