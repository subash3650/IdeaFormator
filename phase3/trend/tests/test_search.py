"""Tests for TrendSearch."""

from __future__ import annotations

from phase3.trend.search import TrendSearch
from phase3.trend.schema import (
    Trend,
    TrendDirection,
    TrendMetrics,
    TrendSubject,
    TrendType,
)


def _make_trend(tid: str, ttype: TrendType = TrendType.GROWING,
                title: str = "", platforms: list[str] | None = None,
                score: float = 0.5, growth: float = 50.0) -> Trend:
    return Trend(
        trend_id=tid,
        title=title or f"Trend {tid}",
        summary="Test summary for trend",
        trend_type=ttype,
        trend_direction=TrendDirection.UP if ttype != TrendType.DECLINING else TrendDirection.DOWN,
        trend_subject=TrendSubject.PROBLEM,
        subject_id=tid,
        subject_label=f"Subject {tid}",
        affected_platforms=platforms or [],
        metrics=TrendMetrics(trend_score=score, growth_pct=growth),
    )


class TestTrendSearch:
    def test_find_by_id(self) -> None:
        trends = [_make_trend("t1"), _make_trend("t2")]
        search = TrendSearch(trends)
        assert search.find_by_id("t1") is not None
        assert search.find_by_id("t3") is None

    def test_find_growing(self) -> None:
        trends = [
            _make_trend("t1", TrendType.GROWING, score=0.8),
            _make_trend("t2", TrendType.STABLE),
        ]
        search = TrendSearch(trends)
        results = search.find_growing()
        assert len(results) == 1

    def test_find_growing_min_score(self) -> None:
        trends = [
            _make_trend("t1", TrendType.GROWING, score=0.3),
            _make_trend("t2", TrendType.GROWING, score=0.8),
        ]
        search = TrendSearch(trends)
        results = search.find_growing(min_score=0.5)
        assert len(results) == 1

    def test_find_declining(self) -> None:
        trends = [
            _make_trend("t1", TrendType.DECLINING),
            _make_trend("t2", TrendType.GROWING),
        ]
        search = TrendSearch(trends)
        results = search.find_declining()
        assert len(results) == 1

    def test_find_emerging(self) -> None:
        trends = [
            _make_trend("t1", TrendType.EMERGING, growth=100.0),
            _make_trend("t2", TrendType.GROWING),
        ]
        search = TrendSearch(trends)
        results = search.find_emerging()
        assert len(results) == 1

    def test_find_anomalies(self) -> None:
        trends = [
            _make_trend("t1", TrendType.ANOMALY),
            _make_trend("t2", TrendType.GROWING),
        ]
        search = TrendSearch(trends)
        results = search.find_anomalies()
        assert len(results) == 1

    def test_find_by_platform(self) -> None:
        t = _make_trend("t1", platforms=["iOS", "Android"])
        search = TrendSearch([t])
        results = search.find_by_platform("iOS")
        assert len(results) == 1

    def test_find_cross_platform(self) -> None:
        t1 = _make_trend("t1", platforms=["iOS", "Android"])
        t2 = _make_trend("t2", platforms=["iOS"])
        search = TrendSearch([t1, t2])
        results = search.find_cross_platform()
        assert len(results) == 1

    def test_search_text(self) -> None:
        t1 = _make_trend("t1", title="Performance Issues")
        t2 = _make_trend("t2", title="Battery Drain")
        search = TrendSearch([t1, t2])
        results = search.search_text("performance")
        assert len(results) == 1
        assert results[0].trend_id == "t1"

    def test_search_text_no_match(self) -> None:
        t = _make_trend("t1", title="Test")
        search = TrendSearch([t])
        results = search.search_text("nonexistent")
        assert results == []

    def test_search_text_empty(self) -> None:
        t = _make_trend("t1", title="Test")
        search = TrendSearch([t])
        results = search.search_text("")
        assert results == []
