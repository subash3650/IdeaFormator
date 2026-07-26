"""Trend score providers for the Trend Intelligence Engine.

Eager-imports all built-in providers so their decorators fire.
"""

from __future__ import annotations

from phase3.trend.providers.base import TrendScoreProvider
from phase3.trend.providers.registry import (
    available_trend_score_providers,
    create_trend_score_provider,
    get_trend_score_provider_class,
    register_trend_score_provider,
    sorted_trend_score_providers,
)

# Eager-import all built-in scoring providers so decorators fire
from phase3.trend.providers.growth import GrowthScoreProvider
from phase3.trend.providers.velocity import VelocityScoreProvider
from phase3.trend.providers.momentum import MomentumScoreProvider
from phase3.trend.providers.confidence import ConfidenceScoreProvider
from phase3.trend.providers.trend_score import TrendScoreCompositeProvider

__all__ = [
    "TrendScoreProvider",
    "register_trend_score_provider",
    "get_trend_score_provider_class",
    "create_trend_score_provider",
    "available_trend_score_providers",
    "sorted_trend_score_providers",
    "GrowthScoreProvider",
    "VelocityScoreProvider",
    "MomentumScoreProvider",
    "ConfidenceScoreProvider",
    "TrendScoreCompositeProvider",
]
