"""Phase 3, Module 4 — Trend Intelligence Engine."""

from __future__ import annotations

from phase3.trend.schema import (
    Trend,
    TrendMetadata,
    TrendOutput,
    TrendMetrics,
    TrendScoringBreakdown,
    TrendScoreWeights,
    TrendSnapshot,
    TrendCorrelation,
    TrendType,
    TrendStatus,
    TrendDirection,
    TrendSubject,
    CorrelationType,
)
from phase3.trend.config import TrendConfig, load_trend_config
from phase3.trend.store import TrendStore

__all__ = [
    "TrendType",
    "TrendStatus",
    "TrendDirection",
    "TrendSubject",
    "CorrelationType",
    "TrendMetrics",
    "TrendScoringBreakdown",
    "TrendScoreWeights",
    "TrendSnapshot",
    "TrendCorrelation",
    "Trend",
    "TrendMetadata",
    "TrendOutput",
    "TrendConfig",
    "load_trend_config",
    "TrendStore",
]

__version__ = "1.0.0"
