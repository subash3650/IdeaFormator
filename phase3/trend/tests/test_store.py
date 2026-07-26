"""Tests for TrendStore."""

from __future__ import annotations

from pathlib import Path

from phase3.trend.schema import (
    Trend,
    TrendDirection,
    TrendMetadata,
    TrendMetrics,
    TrendScoringBreakdown,
    TrendSubject,
    TrendType,
)
from phase3.trend.store import TrendStore


class TestTrendStore:
    def test_save_and_load_trends(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        trends = [
            Trend(
                trend_id="t1", title="T1", summary="S1",
                trend_type=TrendType.GROWING,
                trend_direction=TrendDirection.UP,
                trend_subject=TrendSubject.PROBLEM,
                subject_id="p1", subject_label="P1",
            ),
            Trend(
                trend_id="t2", title="T2", summary="S2",
                trend_type=TrendType.DECLINING,
                trend_direction=TrendDirection.DOWN,
                trend_subject=TrendSubject.OPPORTUNITY,
                subject_id="o1", subject_label="O1",
            ),
        ]
        store.save_trends(trends, "run1")
        loaded = store.load_trends()
        assert len(loaded) == 2
        ids = {t.trend_id for t in loaded}
        assert "t1" in ids
        assert "t2" in ids

    def test_save_and_load_empty(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        store.save_trends([], "run1")
        loaded = store.load_trends()
        assert loaded == []

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path / "nonexistent")
        loaded = store.load_trends()
        assert loaded == []

    def test_save_and_load_metadata(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        meta = TrendMetadata(run_id="run1", total_trends=5)
        store.save_metadata(meta)
        loaded = store.load_metadata()
        assert loaded is not None
        assert loaded.run_id == "run1"
        assert loaded.total_trends == 5

    def test_load_metadata_nonexistent(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        loaded = store.load_metadata()
        assert loaded is None

    def test_save_manifest(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        manifest = {"run_id": "run1", "total": 10}
        path = store.save_manifest(manifest)
        assert path.exists()
        import json
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["run_id"] == "run1"

    def test_checksums(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        store.save_trends([
            Trend(
                trend_id="t1", title="T", summary="S",
                trend_type=TrendType.GROWING,
                trend_direction=TrendDirection.UP,
                trend_subject=TrendSubject.PROBLEM,
                subject_id="p1", subject_label="P1",
            ),
        ], "run1")
        cs = store.checksums()
        assert "trends.parquet" in cs
        assert len(cs["trends.parquet"]) == 16

    def test_checksums_empty(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        cs = store.checksums()
        assert cs == {}

    def test_file_structure(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        store.save_trends([], "run1")
        store.save_metadata(TrendMetadata(run_id="run1"))
        store.save_manifest({"run_id": "run1"})
        assert (store.trend_dir / "trends.parquet").exists()
        assert (store.trend_dir / "trend_metadata.json").exists()
        assert (store.trend_dir / "trend_manifest.json").exists()

    def test_roundtrip_with_metrics(self, tmp_path: Path) -> None:
        metrics = TrendMetrics(growth_pct=25.0, velocity=5.0, confidence=0.9)
        scoring = TrendScoringBreakdown(growth_score=0.8, trend_score=0.75)
        t = Trend(
            trend_id="t1", title="Growing Problem", summary="S",
            trend_type=TrendType.GROWING,
            trend_direction=TrendDirection.UP,
            trend_subject=TrendSubject.PROBLEM,
            subject_id="p1", subject_label="Slow",
            metrics=metrics, scoring=scoring,
            affected_products=["ProductA"],
            affected_platforms=["iOS"],
        )
        store = TrendStore(tmp_path)
        store.save_trends([t], "run1")
        loaded = store.load_trends()
        assert len(loaded) == 1
        assert loaded[0].metrics.growth_pct == 25.0
        assert loaded[0].scoring.trend_score == 0.75
        assert "ProductA" in loaded[0].affected_products
