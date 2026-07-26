"""Tests for Growth, Velocity, and Momentum analyzers."""

from __future__ import annotations

from phase3.trend.growth import GrowthAnalyzer, MomentumAnalyzer, VelocityAnalyzer


class TestGrowthAnalyzer:
    def test_growth_positive(self) -> None:
        g = GrowthAnalyzer()
        result = g.compute(150, 100, 10)
        assert result["growth_pct"] == 50.0
        assert result["velocity"] == 5.0

    def test_growth_negative(self) -> None:
        g = GrowthAnalyzer()
        result = g.compute(50, 100, 5)
        assert result["growth_pct"] == -50.0
        assert result["velocity"] == -10.0

    def test_growth_zero(self) -> None:
        g = GrowthAnalyzer()
        result = g.compute(0, 0, 1)
        assert result["growth_pct"] == 0.0

    def test_growth_from_zero(self) -> None:
        g = GrowthAnalyzer()
        result = g.compute(100, 0, 1)
        assert result["growth_pct"] == 100.0

    def test_velocity_zero_days(self) -> None:
        g = GrowthAnalyzer()
        result = g.compute(100, 50, 0)
        # Should use 0.1 as min days
        assert result["velocity"] == 500.0

    def test_acceleration(self) -> None:
        g = GrowthAnalyzer()
        accel = g.compute_acceleration(10.0, 5.0, 5)
        assert accel == 1.0

    def test_acceleration_zero_days(self) -> None:
        g = GrowthAnalyzer()
        accel = g.compute_acceleration(10, 5, 0)
        assert accel == 0.0

    def test_velocity_only(self) -> None:
        g = GrowthAnalyzer()
        result = g.compute(100, 50, 1)
        assert result["velocity"] == 50.0


class TestVelocityAnalyzer:
    def test_single_point(self) -> None:
        v = VelocityAnalyzer()
        result = v.compute([100], ["2026-01-01T00:00:00"])
        assert result["avg_velocity"] == 0.0
        assert result["momentum"] == 0.0

    def test_two_points(self) -> None:
        v = VelocityAnalyzer()
        result = v.compute(
            [100, 200],
            ["2026-01-01T00:00:00", "2026-01-11T00:00:00"],
        )
        assert result["avg_velocity"] == 10.0
        assert result["peak_velocity"] == 10.0

    def test_multiple_points(self) -> None:
        v = VelocityAnalyzer()
        result = v.compute(
            [100, 150, 250, 400],
            ["2026-01-01T00:00:00", "2026-01-11T00:00:00",
             "2026-01-21T00:00:00", "2026-01-31T00:00:00"],
        )
        assert result["avg_velocity"] == 10.0
        assert result["peak_velocity"] == 15.0
        assert result["momentum"] > 0.0

    def test_decreasing(self) -> None:
        v = VelocityAnalyzer()
        result = v.compute(
            [400, 300, 200],
            ["2026-01-01T00:00:00", "2026-01-11T00:00:00", "2026-01-21T00:00:00"],
        )
        assert result["avg_velocity"] < 0
        assert result["momentum"] < 0


class TestMomentumAnalyzer:
    def test_positive(self) -> None:
        m = MomentumAnalyzer()
        score = m.compute(50.0, 10.0, 5.0, 5)
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_zero(self) -> None:
        m = MomentumAnalyzer()
        score = m.compute(0.0, 0.0, 0.0, 1)
        assert score == 0.01

    def test_max(self) -> None:
        m = MomentumAnalyzer()
        score = m.compute(200.0, 1000.0, 100.0, 10)
        assert score <= 1.0
        assert score > 0.5
