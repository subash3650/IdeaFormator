"""Scoring, business model, and ranking providers for the Opportunity Engine."""

from __future__ import annotations

from phase3.opportunity.providers.base import (
    BusinessModelProvider,
    RankingProvider,
    ScoringProvider,
)
from phase3.opportunity.providers.registry import (
    available_business_model_providers,
    available_ranking_providers,
    available_scoring_providers,
    create_business_model_provider,
    create_ranking_provider,
    create_scoring_provider,
    register_business_model_provider,
    register_ranking_provider,
    register_scoring_provider,
)

# Eager-import all built-in scoring providers so decorators fire
from phase3.opportunity.providers.weighted import WeightedScoreProvider
from phase3.opportunity.providers.market import MarketScoreProvider
from phase3.opportunity.providers.trend import TrendScoreProvider
from phase3.opportunity.providers.competition import CompetitionScoreProvider

# Eager-import all built-in business model providers
from phase3.opportunity.providers.business_model import (
    AIAgentProvider,
    APIProvider,
    B2BPlatformProvider,
    ChromeExtensionProvider,
    ConsumerProductProvider,
    DeveloperToolProvider,
    MarketplaceProvider,
    MobileAppProvider,
    SaaSProvider,
)

__all__ = [
    "ScoringProvider",
    "BusinessModelProvider",
    "RankingProvider",
    "register_scoring_provider",
    "register_business_model_provider",
    "register_ranking_provider",
    "create_scoring_provider",
    "create_business_model_provider",
    "create_ranking_provider",
    "available_scoring_providers",
    "available_business_model_providers",
    "available_ranking_providers",
    "WeightedScoreProvider",
    "MarketScoreProvider",
    "TrendScoreProvider",
    "CompetitionScoreProvider",
    "SaaSProvider",
    "AIAgentProvider",
    "MarketplaceProvider",
    "ChromeExtensionProvider",
    "APIProvider",
    "MobileAppProvider",
    "B2BPlatformProvider",
    "DeveloperToolProvider",
    "ConsumerProductProvider",
]
