"""Tests for TrendDetector."""

from __future__ import annotations

from pathlib import Path

from phase3.trend.config import TrendConfig
from phase3.trend.detector import TrendDetector
from phase3.trend.schema import TrendSnapshot, TrendSubject
from phase3.trend.snapshot import SnapshotDelta
from phase3.trend.timeline import TrendTimeline, TrendTimelinePoint


def _make_snap(sid: str, run_id: str, ts: str) -> TrendSnapshot:
    return TrendSnapshot(snapshot_id=sid, run_id=run_id, timestamp=ts)


class TestTrendDetector:
    def test_empty_delta(self) -> None:
        cfg = TrendConfig(output_dir=Path("/tmp"), min_growth_pct=0.0)
        detector = TrendDetector(cfg)
        prior = _make_snap("s1", "r1", "2026-01-01T00:00:00")
        current = _make_snap("s2", "r2", "2026-02-01T00:00:00")
        delta = SnapshotDelta()
        delta.observation_count_before = 0
        delta.observation_count_after = 0
        timeline = TrendTimeline([
            TrendTimelinePoint(prior),
            TrendTimelinePoint(current),
        ])
        candidates = detector.detect(delta, timeline, prior, current)
        assert candidates == []

    def test_detect_observation_growth(self) -> None:
        cfg = TrendConfig(output_dir=Path("/tmp"), min_growth_pct=0.0)
        detector = TrendDetector(cfg)
        prior = _make_snap("s1", "r1", "2026-01-01T00:00:00")
        current = _make_snap("s2", "r2", "2026-02-01T00:00:00")
        delta = SnapshotDelta()
        delta.observation_count_before = 100
        delta.observation_count_after = 200
        timeline = TrendTimeline([
            TrendTimelinePoint(prior),
            TrendTimelinePoint(current),
        ])
        candidates = detector.detect(delta, timeline, prior, current)
        assert len(candidates) > 0

    def test_detect_with_entity(self) -> None:
        cfg = TrendConfig(output_dir=Path("/tmp"), min_growth_pct=0.0)
        detector = TrendDetector(cfg)
        prior = _make_snap("s1", "r1", "2026-01-01T00:00:00")
        current = _make_snap("s2", "r2", "2026-02-01T00:00:00")
        delta = SnapshotDelta()
        delta.entity_counts_before = {"ProductX": 5}
        delta.entity_counts_after = {"ProductX": 15}
        timeline = TrendTimeline([
            TrendTimelinePoint(prior),
            TrendTimelinePoint(current),
        ])
        candidates = detector.detect(delta, timeline, prior, current)
        assert len(candidates) > 0
        product_candidates = [c for c in candidates if c.subject_id == "ProductX"]
        assert len(product_candidates) > 0

    def test_classify_growing(self) -> None:
        cfg = TrendConfig(output_dir=Path("/tmp"), min_growth_pct=5.0)
        detector = TrendDetector(cfg)
        from phase3.trend.detector import TrendCandidate
        candidate = TrendCandidate(
            subject_id="p1", subject_label="Problem1",
            trend_subject=TrendSubject.PROBLEM,
            growth_pct=50.0,
        )
        ttype, tdir = detector.classify_trend(candidate, TrendTimeline([]))
        assert ttype.value == "growing"
        assert tdir.value == "up"

    def test_classify_declining(self) -> None:
        cfg = TrendConfig(output_dir=Path("/tmp"), min_growth_pct=5.0)
        detector = TrendDetector(cfg)
        from phase3.trend.detector import TrendCandidate
        candidate = TrendCandidate(
            subject_id="p1", subject_label="Problem1",
            trend_subject=TrendSubject.PROBLEM,
            growth_pct=-30.0,
        )
        ttype, tdir = detector.classify_trend(candidate, TrendTimeline([]))
        assert ttype.value == "declining"
        assert tdir.value == "down"

    def test_classify_stable(self) -> None:
        cfg = TrendConfig(output_dir=Path("/tmp"), min_growth_pct=5.0)
        detector = TrendDetector(cfg)
        from phase3.trend.detector import TrendCandidate
        candidate = TrendCandidate(
            subject_id="p1", subject_label="Problem1",
            trend_subject=TrendSubject.PROBLEM,
            growth_pct=1.0,
        )
        ttype, tdir = detector.classify_trend(candidate, TrendTimeline([]))
        assert ttype.value == "stable"
        assert tdir.value == "flat"

    def test_classify_emerging(self) -> None:
        cfg = TrendConfig(output_dir=Path("/tmp"), min_growth_pct=5.0)
        detector = TrendDetector(cfg)
        from phase3.trend.detector import TrendCandidate
        snap = _make_snap("s2", "r2", "2026-02-01T00:00:00")
        candidate = TrendCandidate(
            subject_id="p1", subject_label="New1",
            trend_subject=TrendSubject.PROBLEM,
            prior_value=0, current_value=100,
            current_snapshot=snap,
        )
        ttype, tdir = detector.classify_trend(candidate, TrendTimeline([]))
        assert ttype.value == "emerging"
        assert tdir.value == "up"

    def test_build_metrics(self) -> None:
        cfg = TrendConfig(output_dir=Path("/tmp"), min_growth_pct=0.0)
        detector = TrendDetector(cfg)
        prior = _make_snap("s1", "r1", "2026-01-01T00:00:00")
        current = _make_snap("s2", "r2", "2026-02-01T00:00:00")
        from phase3.trend.detector import TrendCandidate
        candidate = TrendCandidate(
            subject_id="p1", subject_label="P1",
            trend_subject=TrendSubject.PROBLEM,
            prior_value=50, current_value=100,
            prior_snapshot=prior, current_snapshot=current,
        )
        metrics = detector.build_metrics(candidate, days_elapsed=31)
        assert metrics.growth_pct == 100.0
        assert metrics.snapshot_count == 2
        assert metrics.duration_days == 31

    def test_full_detection_pipeline(self) -> None:
        cfg = TrendConfig(output_dir=Path("/tmp"), min_growth_pct=0.0)
        detector = TrendDetector(cfg)
        prior = _make_snap("s1", "r1", "2026-01-01T00:00:00")
        current = _make_snap("s2", "r2", "2026-02-01T00:00:00")
        delta = SnapshotDelta()
        delta.observation_count_before = 500
        delta.observation_count_after = 800
        delta.evidence_count_before = 100
        delta.evidence_count_after = 200
        delta.opportunity_count_before = 5
        delta.opportunity_count_after = 12
        delta.entity_counts_before = {"ProductX": 3, "CompanyY": 2}
        delta.entity_counts_after = {"ProductX": 10, "CompanyY": 2}
        timeline = TrendTimeline([
            TrendTimelinePoint(prior),
            TrendTimelinePoint(current),
        ])
        candidates = detector.detect(delta, timeline, prior, current)
        assert len(candidates) >= 4  # observations, evidence, opportunities, ProductX
