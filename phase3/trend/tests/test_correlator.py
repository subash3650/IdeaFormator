"""Tests for TrendCorrelator."""

from __future__ import annotations

from phase3.trend.correlator import TrendCorrelator
from phase3.trend.schema import (
    Trend,
    TrendDirection,
    TrendMetrics,
    TrendSubject,
    TrendType,
)


def _make_trend(tid: str, platforms: list[str] | None = None,
                products: list[str] | None = None,
                companies: list[str] | None = None,
                technologies: list[str] | None = None) -> Trend:
    return Trend(
        trend_id=tid,
        title=f"Trend {tid}",
        summary="Test",
        trend_type=TrendType.GROWING,
        trend_direction=TrendDirection.UP,
        trend_subject=TrendSubject.PROBLEM,
        subject_id=tid,
        subject_label=f"Subject {tid}",
        affected_platforms=platforms or [],
        affected_products=products or [],
        affected_companies=companies or [],
        affected_technologies=technologies or [],
    )


class TestTrendCorrelator:
    def test_empty(self) -> None:
        correlator = TrendCorrelator()
        result = correlator.correlate([])
        assert result == []

    def test_no_correlations(self) -> None:
        correlator = TrendCorrelator()
        t = _make_trend("t1")
        result = correlator.correlate([t])
        assert len(result[0].correlations) == 0

    def test_cross_platform(self) -> None:
        correlator = TrendCorrelator()
        t1 = _make_trend("t1", platforms=["iOS", "Android", "Web"])
        t2 = _make_trend("t2", platforms=["iOS", "Android", "Desktop"])
        result = correlator.correlate([t1, t2])
        t1_corrs = [c for c in result[0].correlations if c.correlation_type.value == "cross_platform"]
        assert len(t1_corrs) > 0

    def test_product_company_correlation(self) -> None:
        correlator = TrendCorrelator()
        t = _make_trend("t1", products=["ProductA"], companies=["CompanyB"])
        result = correlator.correlate([t])
        company_corrs = [c for c in result[0].correlations if c.correlation_type.value == "company_trend"]
        assert len(company_corrs) > 0

    def test_technology_product_correlation(self) -> None:
        correlator = TrendCorrelator()
        t = _make_trend("t1", products=["ProductA"], technologies=["TechX"])
        result = correlator.correlate([t])
        tech_corrs = [c for c in result[0].correlations if c.correlation_type.value == "problem_technology"]
        assert len(tech_corrs) > 0

    def test_no_cross_platform_single_platform(self) -> None:
        correlator = TrendCorrelator()
        t1 = _make_trend("t1", platforms=["iOS"])
        t2 = _make_trend("t2", platforms=["Android"])
        result = correlator.correlate([t1, t2])
        assert len(result[0].correlations) == 0
