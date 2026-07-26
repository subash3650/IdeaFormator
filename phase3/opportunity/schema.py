"""Pydantic models for the Opportunity Discovery Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OpportunityType(str, Enum):
    SAAS = "saas"
    AI_AGENT = "ai_agent"
    MARKETPLACE = "marketplace"
    CHROME_EXTENSION = "chrome_extension"
    API = "api"
    MOBILE_APP = "mobile_app"
    B2B_PLATFORM = "b2b_platform"
    DEVELOPER_TOOL = "developer_tool"
    CONSUMER_PRODUCT = "consumer_product"


class RecommendationType(str, Enum):
    STRONG_PURSUE = "strong_pursue"
    WORTH_EXPLORING = "worth_exploring"
    NICHE_POTENTIAL = "niche_potential"
    MONITOR = "monitor"
    INSUFFICIENT_DATA = "insufficient_data"


class OpportunityStatus(str, Enum):
    IDENTIFIED = "identified"
    VALIDATED = "validated"
    SCORED = "scored"
    RANKED = "ranked"
    RECOMMENDED = "recommended"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class RankingStrategy(str, Enum):
    COMPOSITE = "composite"
    PAIN_SEVERITY = "pain_severity"
    MARKET_SIZE = "market_size"
    FEASIBILITY = "feasibility"
    CONFIDENCE = "confidence"


class MarketSize(str, Enum):
    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"
    UNKNOWN = "unknown"


class MarketMaturity(str, Enum):
    EMERGING = "emerging"
    GROWING = "growing"
    MATURE = "mature"
    DECLINING = "declining"
    UNKNOWN = "unknown"


class ImplementationComplexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Scoring models
# ---------------------------------------------------------------------------


class ScoreWeights(BaseModel):
    """Configurable weights for composite opportunity scoring."""

    model_config = {"frozen": True, "extra": "forbid"}

    pain_severity: float = Field(default=0.20, ge=0.0, le=1.0)
    frequency: float = Field(default=0.15, ge=0.0, le=1.0)
    trend: float = Field(default=0.10, ge=0.0, le=1.0)
    evidence_count: float = Field(default=0.10, ge=0.0, le=1.0)
    reasoning_confidence: float = Field(default=0.10, ge=0.0, le=1.0)
    cluster_density: float = Field(default=0.08, ge=0.0, le=1.0)
    cross_platform: float = Field(default=0.07, ge=0.0, le=1.0)
    market_coverage: float = Field(default=0.05, ge=0.0, le=1.0)
    competition: float = Field(default=0.05, ge=0.0, le=1.0)
    feasibility: float = Field(default=0.05, ge=0.0, le=1.0)
    novelty: float = Field(default=0.05, ge=0.0, le=1.0)

    def normalize(self) -> ScoreWeights:
        """Return normalized weights summing to 1.0."""
        total = sum(
            [
                self.pain_severity,
                self.frequency,
                self.trend,
                self.evidence_count,
                self.reasoning_confidence,
                self.cluster_density,
                self.cross_platform,
                self.market_coverage,
                self.competition,
                self.feasibility,
                self.novelty,
            ]
        )
        if abs(total - 1.0) < 1e-9:
            return self
        if total == 0:
            return ScoreWeights()
        return ScoreWeights(
            pain_severity=self.pain_severity / total,
            frequency=self.frequency / total,
            trend=self.trend / total,
            evidence_count=self.evidence_count / total,
            reasoning_confidence=self.reasoning_confidence / total,
            cluster_density=self.cluster_density / total,
            cross_platform=self.cross_platform / total,
            market_coverage=self.market_coverage / total,
            competition=self.competition / total,
            feasibility=self.feasibility / total,
            novelty=self.novelty / total,
        )


class ScoringBreakdown(BaseModel):
    """Per-factor scoring breakdown for explainability."""

    model_config = {"frozen": True, "extra": "forbid"}

    pain_severity: float = Field(default=0.0, ge=0.0, le=1.0)
    frequency: float = Field(default=0.0, ge=0.0, le=1.0)
    trend: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cluster_density: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_platform: float = Field(default=0.0, ge=0.0, le=1.0)
    market_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    competition: float = Field(default=0.0, ge=0.0, le=1.0)
    feasibility: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)


class ConfidenceBreakdown(BaseModel):
    """Multi-dimensional confidence breakdown."""

    model_config = {"frozen": True, "extra": "forbid"}

    reasoning_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    graph_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    market_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    computation_method: str = Field(default="weighted_average")


# ---------------------------------------------------------------------------
# Core Opportunity model
# ---------------------------------------------------------------------------


class Opportunity(BaseModel):
    """A ranked business opportunity derived from pain intelligence."""

    model_config = {"frozen": True, "extra": "forbid"}

    opportunity_id: str = Field(description="Deterministic SHA-256 identifier")
    title: str = Field(description="Human-readable title, max 200 characters")
    summary: str = Field(description="1-3 sentence description of the opportunity")

    root_problem: str = Field(description="Core pain point identifier")

    supporting_evidence: list[str] = Field(
        default_factory=list, description="Evidence node IDs that support this opportunity"
    )
    reasoning_chain_ids: list[str] = Field(
        default_factory=list, description="Reasoning chain IDs that support this opportunity"
    )
    cluster_ids: list[str] = Field(
        default_factory=list, description="Semantic cluster IDs associated with this opportunity"
    )
    kg_node_ids: list[str] = Field(
        default_factory=list, description="Knowledge graph node IDs associated with this opportunity"
    )

    affected_products: list[str] = Field(default_factory=list)
    affected_companies: list[str] = Field(default_factory=list)
    affected_technologies: list[str] = Field(default_factory=list)

    estimated_market_size: MarketSize = Field(default=MarketSize.UNKNOWN)
    market_maturity: MarketMaturity = Field(default=MarketMaturity.UNKNOWN)
    implementation_complexity: ImplementationComplexity = Field(
        default=ImplementationComplexity.UNKNOWN
    )
    time_to_market: str = Field(default="")
    investment_required: str = Field(default="")
    estimated_revenue: str = Field(default="")
    recurring_revenue_potential: bool = Field(default=False)
    strategic_fit: str = Field(default="")

    pain_severity: float = Field(default=0.0, ge=0.0, le=1.0)
    frequency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    trend_score: float = Field(default=0.0, ge=0.0, le=1.0)
    competition_score: float = Field(default=0.0, ge=0.0, le=1.0)
    feasibility_score: float = Field(default=0.0, ge=0.0, le=1.0)

    opportunity_score: float = Field(default=0.0, ge=0.0, le=1.0)

    scoring_breakdown: ScoringBreakdown = Field(default_factory=ScoringBreakdown)
    confidence: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)

    recommendation_type: RecommendationType = Field(default=RecommendationType.INSUFFICIENT_DATA)
    suggested_solution: str = Field(default="")
    suggested_business_model: OpportunityType = Field(default=OpportunityType.SAAS)

    status: OpportunityStatus = Field(default=OpportunityStatus.IDENTIFIED)
    rank: int = Field(default=0, ge=0)

    created_at: str = Field(default_factory=_now_iso)
    pipeline_version: str = Field(default="1.0")
    schema_version: str = Field(default="1.0")


# ---------------------------------------------------------------------------
# Metadata and output container
# ---------------------------------------------------------------------------


class OpportunityMetadata(BaseModel):
    """Run-level metadata for an opportunity discovery session."""

    model_config = {"frozen": True, "extra": "forbid"}

    run_id: str = Field(description="Opportunity discovery run ID")
    reasoning_run_id: str = Field(default="", description="Reasoning engine run ID used as input")
    kg_run_id: str = Field(default="", description="Knowledge graph run ID used as input")

    total_candidates: int = Field(default=0, ge=0)
    total_opportunities: int = Field(default=0, ge=0)
    scored_opportunities: int = Field(default=0, ge=0)
    ranked_opportunities: int = Field(default=0, ge=0)

    scoring_providers_used: list[str] = Field(default_factory=list)
    business_model_providers_used: list[str] = Field(default_factory=list)

    recommendation_distribution: dict[str, int] = Field(default_factory=dict)
    business_model_distribution: dict[str, int] = Field(default_factory=dict)

    avg_opportunity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    min_opportunity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_opportunity_score: float = Field(default=0.0, ge=0.0, le=1.0)

    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    cache_hit: bool = False

    created_at: str = Field(default_factory=_now_iso)
    pipeline_version: str = Field(default="1.0")
    schema_version: str = Field(default="1.0")


class OpportunityOutput(BaseModel):
    """Aggregate output container for opportunity discovery results."""

    model_config = {"frozen": True, "extra": "forbid"}

    opportunities: list[Opportunity] = Field(default_factory=list)
    metadata: OpportunityMetadata | None = None


# ---------------------------------------------------------------------------
# Provider marker types (used by registry, not ABCs)
# ---------------------------------------------------------------------------


class ScoringProvider(str, Enum):
    SCORING = "scoring"


class BusinessModelProvider(str, Enum):
    BUSINESS_MODEL = "business_model"


class RankingProvider(str, Enum):
    RANKING = "ranking"
