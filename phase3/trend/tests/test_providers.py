"""Tests for trend score providers and their registry."""

from __future__ import annotations

import pytest

# Eager-import providers so decorators fire
import phase3.trend.providers  # noqa: F401

from phase3.trend.providers.base import TrendScoreProvider
from phase3.trend.providers.registry import (
    available_trend_score_providers,
    create_trend_score_provider,
    get_trend_score_provider_class,
    register_trend_score_provider,
    sorted_trend_score_providers,
)
from phase3.trend.schema import TrendScoringBreakdown


class TestTrendScoreProviderRegistry:
    def test_available_providers(self) -> None:
        providers = available_trend_score_providers()
        assert "growth" in providers
        assert "velocity" in providers
        assert "momentum" in providers
        assert "confidence" in providers
        assert "trend_score" in providers

    def test_get_provider_class(self) -> None:
        cls = get_trend_score_provider_class("growth")
        from phase3.trend.providers.growth import GrowthScoreProvider
        assert cls is GrowthScoreProvider

    def test_get_provider_class_invalid(self) -> None:
        with pytest.raises(KeyError):
            get_trend_score_provider_class("nonexistent")

    def test_create_provider(self) -> None:
        provider = create_trend_score_provider("growth")
        assert provider.name == "growth"
        assert provider.priority == 100

    def test_sorted_order(self) -> None:
        names = sorted_trend_score_providers()
        assert names.index("growth") < names.index("trend_score")

    def test_register_custom_provider(self) -> None:
        @register_trend_score_provider(name="custom_test", priority=50)
        class CustomProvider(TrendScoreProvider):
            @property
            def name(self) -> str:
                return "custom_test"
            @property
            def priority(self) -> int:
                return 50
            def score(self, candidate: dict, context: dict) -> TrendScoringBreakdown:
                return TrendScoringBreakdown()

        assert "custom_test" in available_trend_score_providers()
        provider = create_trend_score_provider("custom_test")
        assert provider.name == "custom_test"


class TestGrowthScoreProvider:
    def test_score_positive(self) -> None:
        provider = create_trend_score_provider("growth")
        result = provider.score({"growth_pct": 50.0}, {})
        assert result.growth_score == 0.5

    def test_score_negative(self) -> None:
        provider = create_trend_score_provider("growth")
        result = provider.score({"growth_pct": -30.0}, {})
        assert result.growth_score == 0.3

    def test_score_zero(self) -> None:
        provider = create_trend_score_provider("growth")
        result = provider.score({"growth_pct": 0.0}, {})
        assert result.growth_score == 0.0


class TestVelocityScoreProvider:
    def test_score(self) -> None:
        provider = create_trend_score_provider("velocity")
        result = provider.score({"velocity": 500.0}, {})
        assert result.velocity_score == 0.5

    def test_score_zero(self) -> None:
        provider = create_trend_score_provider("velocity")
        result = provider.score({"velocity": 0.0}, {})
        assert result.velocity_score == 0.0


class TestMomentumScoreProvider:
    def test_score(self) -> None:
        provider = create_trend_score_provider("momentum")
        result = provider.score({"momentum": 0.75}, {})
        assert result.momentum_score == 0.75

    def test_score_zero(self) -> None:
        provider = create_trend_score_provider("momentum")
        result = provider.score({"momentum": 0.0}, {})
        assert result.momentum_score == 0.0


class TestConfidenceScoreProvider:
    def test_score_high(self) -> None:
        provider = create_trend_score_provider("confidence")
        result = provider.score(
            {"snapshot_count": 10, "confidence": 1.0, "total_observations": 1000},
            {},
        )
        assert result.confidence_score > 0.5

    def test_score_low(self) -> None:
        provider = create_trend_score_provider("confidence")
        result = provider.score(
            {"snapshot_count": 1, "confidence": 0.0, "total_observations": 0},
            {},
        )
        assert result.confidence_score == 0.03


class TestTrendScoreCompositeProvider:
    def test_score_composite(self) -> None:
        provider = create_trend_score_provider("trend_score")
        context = {"score_weights": {"growth": 0.5, "velocity": 0.5}}
        candidate = {"growth_score": 0.8, "velocity_score": 0.6}
        result = provider.score(candidate, context)
        assert result.trend_score == 0.7

    def test_score_with_defaults(self) -> None:
        provider = create_trend_score_provider("trend_score")
        result = provider.score({}, {})
        assert result.trend_score == 0.0

    def test_score_with_all_dimensions(self) -> None:
        provider = create_trend_score_provider("trend_score")
        context = {"score_weights": {"growth": 0.3, "velocity": 0.2, "momentum": 0.15,
                                      "confidence": 0.1, "seasonality": 0.05,
                                      "anomaly": 0.1, "cross_platform": 0.1}}
        candidate = {"growth_score": 1.0, "velocity_score": 1.0, "momentum_score": 1.0,
                     "confidence_score": 1.0, "seasonality_score": 1.0,
                     "anomaly_score": 1.0, "cross_platform_score": 1.0}
        result = provider.score(candidate, context)
        assert abs(result.trend_score - 1.0) < 1e-6

    def test_all_providers_return_breakdown(self) -> None:
        context = {"score_weights": {}, "max_snapshots": 10}
        candidate = {"growth_pct": 10.0, "velocity": 5.0, "momentum": 0.5,
                     "confidence": 0.8, "snapshot_count": 5, "total_observations": 500,
                     "growth_score": 0.5, "velocity_score": 0.5,
                     "momentum_score": 0.5, "confidence_score": 0.5,
                     "seasonality_score": 0.5, "anomaly_score": 0.5,
                     "cross_platform_score": 0.5}
        for name in available_trend_score_providers():
            provider = create_trend_score_provider(name)
            result = provider.score(candidate, context)
            assert isinstance(result, TrendScoringBreakdown)
            for field in TrendScoringBreakdown.model_fields:
                val = getattr(result, field)
                assert 0.0 <= val <= 1.0, f"{name}.{field} = {val}"
