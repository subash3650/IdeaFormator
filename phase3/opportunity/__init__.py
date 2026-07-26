"""Phase 3, Module 3 — Opportunity Discovery Engine."""

from __future__ import annotations

from phase3.opportunity.schema import (
    BusinessModelProvider,
    Opportunity,
    OpportunityMetadata,
    OpportunityOutput,
    OpportunityStatus,
    OpportunityType,
    RankingProvider,
    RankingStrategy,
    RecommendationType,
    ScoreWeights,
    ScoringBreakdown,
    ScoringProvider,
)
from phase3.opportunity.config import OpportunityConfig, load_opportunity_config
from phase3.opportunity.store import OpportunityStore

__all__ = [
    "OpportunityType",
    "RecommendationType",
    "OpportunityStatus",
    "RankingStrategy",
    "BusinessModelProvider",
    "ScoringProvider",
    "RankingProvider",
    "ScoreWeights",
    "ScoringBreakdown",
    "Opportunity",
    "OpportunityMetadata",
    "OpportunityOutput",
    "OpportunityConfig",
    "load_opportunity_config",
    "OpportunityStore",
]

__version__ = "1.0.0"
