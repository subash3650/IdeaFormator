"""Tests for TrendExporter."""

from __future__ import annotations

from pathlib import Path

from phase3.trend.schema import (
    Trend,
    TrendDirection,
    TrendMetadata,
    TrendMetrics,
    TrendSubject,
    TrendType,
)
from phase3.trend.store import TrendStore
from phase3.trend.exporter import TrendExporter


class TestTrendExporter:
    def test_export_report(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        store.save_trends([
            _make_trend("t1"),
        ], "run1")
        store.save_metadata(TrendMetadata(run_id="run1"))
        exporter = TrendExporter(store)
        path = exporter.export_report()
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "total_trends" in data

    def test_export_statistics(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        store.save_trends([
            _make_trend("t1", TrendType.GROWING, score=0.85),
        ], "run1")
        exporter = TrendExporter(store)
        path = exporter.export_statistics()
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "trend_count" in data

    def test_export_dashboard(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        store.save_trends([
            _make_trend("t1", TrendType.GROWING, score=0.9),
            _make_trend("t2", TrendType.DECLINING, score=0.7),
        ], "run1")
        exporter = TrendExporter(store)
        path = exporter.export_dashboard()
        assert path.exists()

    def test_export_dashboard_text(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        store.save_trends([
            _make_trend("t1"),
        ], "run1")
        exporter = TrendExporter(store)
        path = exporter.export_dashboard_text()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "TREND INTELLIGENCE" in content

    def test_export_summary(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        store.save_trends([
            _make_trend("t1"),
        ], "run1")
        store.save_metadata(TrendMetadata(run_id="run1"))
        exporter = TrendExporter(store)
        path = exporter.export_summary()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Trend Intelligence Summary" in content

    def test_export_csv(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        store.save_trends([
            _make_trend("t1"),
        ], "run1")
        exporter = TrendExporter(store)
        path = exporter.export_csv()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "trend_id" in content

    def test_export_empty(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        exporter = TrendExporter(store)
        path = exporter.export_report()
        assert path.exists()
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["total_trends"] == 0

    def test_all_exports(self, tmp_path: Path) -> None:
        store = TrendStore(tmp_path)
        store.save_trends([
            _make_trend("t1"),
        ], "run1")
        store.save_metadata(TrendMetadata(run_id="run1"))
        exporter = TrendExporter(store)
        exports = exporter.export_all()
        for name, path in exports.items():
            assert path.exists(), f"{name} not found at {path}"


def _make_trend(tid: str, ttype: TrendType = TrendType.GROWING,
                score: float = 0.5) -> Trend:
    return Trend(
        trend_id=tid,
        title=f"Trend {tid}",
        summary="Test",
        trend_type=ttype,
        trend_direction=TrendDirection.UP,
        trend_subject=TrendSubject.PROBLEM,
        subject_id=tid,
        subject_label=f"Subject {tid}",
        metrics=TrendMetrics(trend_score=score, growth_pct=50.0),
    )
