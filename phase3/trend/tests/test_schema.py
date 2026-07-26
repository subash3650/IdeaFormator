"""Tests for Pydantic models in the Trend Intelligence Engine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from phase3.trend.schema import (
    CorrelationType,
    Trend,
    TrendCorrelation,
    TrendDirection,
    TrendMetadata,
    TrendMetrics,
    TrendOutput,
    TrendScoringBreakdown,
    TrendScoreWeights,
    TrendSnapshot,
    TrendStatus,
    TrendSubject,
    TrendType,
)


class TestEnums:
    def test_trend_type_values(self) -> None:
        assert TrendType.GROWING.value == "growing"
        assert TrendType.DECLINING.value == "declining"

    def test_trend_status_lifecycle(self) -> None:
        assert TrendStatus.IDENTIFIED.value == "identified"
        assert TrendStatus.CONFIRMED.value == "confirmed"

    def test_trend_direction_values(self) -> None:
        assert TrendDirection.UP.value == "up"
        assert TrendDirection.CYCLICAL.value == "cyclical"

    def test_trend_subject_values(self) -> None:
        assert TrendSubject.PROBLEM.value == "problem"
        assert TrendSubject.OPPORTUNITY.value == "opportunity"

    def test_correlation_type_values(self) -> None:
        assert CorrelationType.CROSS_PLATFORM.value == "cross_platform"


class TestTrendScoreWeights:
    def test_defaults(self) -> None:
        w = TrendScoreWeights()
        assert w.growth == 0.30
        assert w.velocity == 0.20

    def test_frozen(self) -> None:
        w = TrendScoreWeights()
        with pytest.raises(ValidationError):
            w.growth = 0.5

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            TrendScoreWeights(unknown=1.0)

    def test_normalize(self) -> None:
        w = TrendScoreWeights(growth=0.5, velocity=0.5)
        n = w.normalize()
        total = n.growth + n.velocity + n.momentum + n.confidence
        total += n.seasonality + n.anomaly + n.cross_platform
        assert abs(total - 1.0) < 1e-6

    def test_normalize_zero(self) -> None:
        w = TrendScoreWeights(growth=0.0, velocity=0.0, momentum=0.0,
                              confidence=0.0, seasonality=0.0,
                              anomaly=0.0, cross_platform=0.0)
        n = w.normalize()
        assert n.growth == 0.30

    def test_range_validation(self) -> None:
        with pytest.raises(ValidationError):
            TrendScoreWeights(growth=1.5)


class TestTrendMetrics:
    def test_defaults(self) -> None:
        m = TrendMetrics()
        assert m.growth_pct == 0.0
        assert m.velocity == 0.0
        assert m.confidence == 0.0

    def test_frozen(self) -> None:
        m = TrendMetrics()
        with pytest.raises(ValidationError):
            m.growth_pct = 10.0

    def test_range(self) -> None:
        with pytest.raises(ValidationError):
            TrendMetrics(growth_pct=-1001.0)
        with pytest.raises(ValidationError):
            TrendMetrics(confidence=1.5)


class TestTrendScoringBreakdown:
    def test_defaults(self) -> None:
        sb = TrendScoringBreakdown()
        assert sb.growth_score == 0.0
        assert sb.trend_score == 0.0

    def test_frozen(self) -> None:
        sb = TrendScoringBreakdown()
        with pytest.raises(ValidationError):
            sb.growth_score = 0.5

    def test_range(self) -> None:
        with pytest.raises(ValidationError):
            TrendScoringBreakdown(growth_score=-0.1)
        with pytest.raises(ValidationError):
            TrendScoringBreakdown(growth_score=1.5)


class TestTrendSnapshot:
    def test_minimal(self) -> None:
        s = TrendSnapshot(snapshot_id="s1", run_id="run1", timestamp="2026-01-01T00:00:00")
        assert s.snapshot_id == "s1"
        assert s.run_id == "run1"

    def test_frozen(self) -> None:
        s = TrendSnapshot(snapshot_id="s1", run_id="run1", timestamp="2026-01-01T00:00:00")
        with pytest.raises(ValidationError):
            s.run_id = "run2"


class TestTrendCorrelation:
    def test_minimal(self) -> None:
        c = TrendCorrelation(
            correlation_id="c1", trend_id="t1",
            related_entity_id="e1",
            correlation_type=CorrelationType.CROSS_PLATFORM,
            correlation_strength=0.8,
        )
        assert c.correlation_id == "c1"
        assert c.correlation_strength == 0.8

    def test_frozen(self) -> None:
        c = TrendCorrelation(
            correlation_id="c1", trend_id="t1",
            related_entity_id="e1",
            correlation_type=CorrelationType.CROSS_PLATFORM,
            correlation_strength=0.8,
        )
        with pytest.raises(ValidationError):
            c.correlation_strength = 0.5


class TestTrend:
    def test_minimal(self) -> None:
        t = Trend(
            trend_id="t1",
            title="Growing Problem",
            summary="Users reporting more issues",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1",
            subject_label="Slow Performance",
        )
        assert t.trend_id == "t1"
        assert t.status == TrendStatus.IDENTIFIED
        assert t.rank == 0

    def test_frozen(self) -> None:
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
        )
        with pytest.raises(ValidationError):
            t.title = "New Title"

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            Trend(
                trend_id="t1", title="T", summary="S",
                trend_type=TrendType.GROWING,
                trend_direction=TrendDirection.UP,
                trend_subject=TrendSubject.PROBLEM,
                subject_id="p1", subject_label="P1",
                unknown_field="x",
            )

    def test_full(self) -> None:
        metrics = TrendMetrics(growth_pct=15.0, velocity=10.0, confidence=0.8)
        scoring = TrendScoringBreakdown(growth_score=0.9, trend_score=0.85)
        t = Trend(
            trend_id="t1",
            title="Growing Problem",
            summary="Users reporting more issues",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1",
            subject_label="Slow Performance",
            snapshot_ids=["s1", "s2"],
            metrics=metrics,
            scoring=scoring,
            affected_products=["ProductA"],
            affected_platforms=["iOS", "Android"],
            status=TrendStatus.CONFIRMED,
            rank=1,
        )
        assert t.metrics.growth_pct == 15.0
        assert t.scoring.trend_score == 0.85
        assert "ProductA" in t.affected_products
        assert t.status == TrendStatus.CONFIRMED
        assert t.rank == 1

    def test_metrics_default(self) -> None:
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
        )
        assert t.metrics.growth_pct == 0.0

    def test_serialization_roundtrip(self) -> None:
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
            metrics=TrendMetrics(growth_pct=10.0),
        )
        data = t.model_dump(mode="json")
        restored = Trend(**data)
        assert restored.trend_id == t.trend_id
        assert restored.metrics.growth_pct == 10.0

    def test_status_default(self) -> None:
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
        )
        assert t.status == TrendStatus.IDENTIFIED

    def test_rank_default(self) -> None:
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
        )
        assert t.rank == 0

    def test_with_correlations(self) -> None:
        corr = TrendCorrelation(
            correlation_id="c1", trend_id="t1",
            related_entity_id="e1",
            correlation_type=CorrelationType.CROSS_PLATFORM,
            correlation_strength=0.9,
        )
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
            correlations=[corr],
        )
        assert len(t.correlations) == 1
        assert t.correlations[0].correlation_strength == 0.9


class TestTrendMetadata:
    def test_minimal(self) -> None:
        m = TrendMetadata(run_id="run1")
        assert m.run_id == "run1"
        assert m.total_trends == 0

    def test_frozen(self) -> None:
        m = TrendMetadata(run_id="run1")
        with pytest.raises(ValidationError):
            m.total_trends = 5


class TestTrendOutput:
    def test_defaults(self) -> None:
        o = TrendOutput()
        assert o.trends == []
        assert o.metadata is None

    def test_with_data(self) -> None:
        t = Trend(
            trend_id="t1", title="T", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="P1",
        )
        meta = TrendMetadata(run_id="run1")
        o = TrendOutput(trends=[t], metadata=meta)
        assert len(o.trends) == 1
        assert o.metadata.run_id == "run1"
