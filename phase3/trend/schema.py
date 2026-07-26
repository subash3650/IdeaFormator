"""Pydantic models for the Trend Intelligence Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrendType(str, Enum):
    GROWING = "growing"
    DECLINING = "declining"
    EMERGING = "emerging"
    STABLE = "stable"
    SEASONAL = "seasonal"
    SPIKE = "spike"
    ANOMALY = "anomaly"
    RECURRING = "recurring"
    CROSS_PLATFORM = "cross_platform"


class TrendStatus(str, Enum):
    IDENTIFIED = "identified"
    CONFIRMED = "confirmed"
    MONITORING = "monitoring"
    DECAYING = "decaying"
    ARCHIVED = "archived"


class TrendDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    CYCLICAL = "cyclical"


class TrendSubject(str, Enum):
    PROBLEM = "problem"
    OPPORTUNITY = "opportunity"
    TECHNOLOGY = "technology"
    COMPANY = "company"
    PRODUCT = "product"
    PLATFORM = "platform"
    SIGNAL = "signal"
    CLUSTER = "cluster"
    CATEGORY = "category"
    FEATURE = "feature"


class CorrelationType(str, Enum):
    PROBLEM_OPPORTUNITY = "problem_opportunity"
    PROBLEM_TECHNOLOGY = "problem_technology"
    PROBLEM_COMPANY = "problem_company"
    OPPORTUNITY_TREND = "opportunity_trend"
    CLUSTER_TREND = "cluster_trend"
    REASONING_TREND = "reasoning_trend"
    CROSS_PLATFORM = "cross_platform"
    COMPANY_TREND = "company_trend"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Trend Metrics
# ---------------------------------------------------------------------------


class TrendMetrics(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    growth_pct: float = Field(default=0.0, ge=-1000.0, le=100000.0)
    velocity: float = Field(default=0.0)
    acceleration: float = Field(default=0.0)
    momentum: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    duration_days: int = Field(default=0, ge=0)
    first_seen: str = Field(default="")
    last_seen: str = Field(default="")
    peak_value: float = Field(default=0.0)
    peak_date: str = Field(default="")
    avg_frequency: float = Field(default=0.0, ge=0.0)
    moving_avg: float = Field(default=0.0)
    trend_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_observations: int = Field(default=0, ge=0)
    snapshot_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Scoring breakdown
# ---------------------------------------------------------------------------


class TrendScoringBreakdown(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    growth_score: float = Field(default=0.0, ge=0.0, le=1.0)
    velocity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    momentum_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    seasonality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_platform_score: float = Field(default=0.0, ge=0.0, le=1.0)
    trend_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------


class TrendScoreWeights(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    growth: float = Field(default=0.30, ge=0.0, le=1.0)
    velocity: float = Field(default=0.20, ge=0.0, le=1.0)
    momentum: float = Field(default=0.15, ge=0.0, le=1.0)
    confidence: float = Field(default=0.10, ge=0.0, le=1.0)
    seasonality: float = Field(default=0.05, ge=0.0, le=1.0)
    anomaly: float = Field(default=0.10, ge=0.0, le=1.0)
    cross_platform: float = Field(default=0.10, ge=0.0, le=1.0)

    def normalize(self) -> TrendScoreWeights:
        total = self.growth + self.velocity + self.momentum + self.confidence
        total += self.seasonality + self.anomaly + self.cross_platform
        if total == 0.0:
            return TrendScoreWeights()
        factor = 1.0 / total
        return TrendScoreWeights(
            growth=round(self.growth * factor, 4),
            velocity=round(self.velocity * factor, 4),
            momentum=round(self.momentum * factor, 4),
            confidence=round(self.confidence * factor, 4),
            seasonality=round(self.seasonality * factor, 4),
            anomaly=round(self.anomaly * factor, 4),
            cross_platform=round(self.cross_platform * factor, 4),
        )


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TrendSnapshot(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    snapshot_id: str = Field(description="Deterministic SHA-256 identifier")
    run_id: str = Field(description="Pipeline run ID")
    timestamp: str = Field(description="ISO-8601 creation timestamp")
    observation_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    signal_count: int = Field(default=0, ge=0)
    opportunity_count: int = Field(default=0, ge=0)
    asset_checksums: dict[str, str] = Field(default_factory=dict)
    asset_sizes: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


class TrendCorrelation(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    correlation_id: str = Field(description="Deterministic SHA-256 identifier")
    trend_id: str = Field(description="Trend ID this correlation belongs to")
    related_entity_id: str = Field(description="ID of the related entity")
    correlation_type: CorrelationType = Field(description="Type of correlation")
    correlation_strength: float = Field(ge=0.0, le=1.0)
    correlation_sign: str = Field(default="positive")
    description: str = Field(default="")


# ---------------------------------------------------------------------------
# Main Trend model
# ---------------------------------------------------------------------------


class Trend(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    trend_id: str = Field(description="Deterministic SHA-256 identifier")
    title: str = Field(description="Human-readable title")
    summary: str = Field(description="1-3 sentence description")
    trend_type: TrendType = Field(description="Classification of trend behavior")
    trend_direction: TrendDirection = Field(description="Direction of change")
    trend_subject: TrendSubject = Field(description="Type of entity being tracked")
    subject_id: str = Field(description="Entity/concept ID being tracked")
    subject_label: str = Field(description="Human-readable entity label")

    # Snapshot references
    snapshot_ids: list[str] = Field(default_factory=list)
    first_snapshot_id: str = Field(default="")
    last_snapshot_id: str = Field(default="")
    prior_snapshot_id: str = Field(default="")

    # Metrics
    metrics: TrendMetrics = Field(default_factory=TrendMetrics)
    scoring: TrendScoringBreakdown = Field(default_factory=TrendScoringBreakdown)
    correlations: list[TrendCorrelation] = Field(default_factory=list)

    # Entity associations
    affected_products: list[str] = Field(default_factory=list)
    affected_companies: list[str] = Field(default_factory=list)
    affected_technologies: list[str] = Field(default_factory=list)
    affected_platforms: list[str] = Field(default_factory=list)
    affected_categories: list[str] = Field(default_factory=list)
    affected_features: list[str] = Field(default_factory=list)

    # Lifecycle
    status: TrendStatus = TrendStatus.IDENTIFIED
    rank: int = Field(default=0, ge=0)

    created_at: str = Field(default_factory=_now_iso)
    pipeline_version: str = Field(default="1.0")
    schema_version: str = Field(default="1.0")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TrendMetadata(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    run_id: str = Field(description="Trend analysis run ID")
    snapshot_count: int = Field(default=0, ge=0)
    total_trends: int = Field(default=0, ge=0)
    cache_hit: bool = Field(default=False)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    first_snapshot_id: str = Field(default="")
    last_snapshot_id: str = Field(default="")
    scoring_providers_used: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class TrendOutput(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    trends: list[Trend] = Field(default_factory=list)
    metadata: TrendMetadata | None = None
    pipeline_version: str = Field(default="1.0")
