"""Tests for scoring, business model, and ranking providers."""

from __future__ import annotations

import pytest

from phase3.opportunity.providers.base import BusinessModelProvider, ScoringProvider
from phase3.opportunity.providers.registry import (
    available_business_model_providers,
    available_scoring_providers,
    create_business_model_provider,
    create_scoring_provider,
    get_scoring_provider_class,
    register_scoring_provider,
    sorted_scoring_providers,
)
from phase3.opportunity.schema import Opportunity, OpportunityType, ScoringBreakdown


class TestScoringProviderRegistry:
    def test_available_providers(self) -> None:
        providers = available_scoring_providers()
        assert "weighted" in providers
        assert "market" in providers
        assert "trend" in providers
        assert "competition" in providers

    def test_get_provider_class(self) -> None:
        cls = get_scoring_provider_class("weighted")
        from phase3.opportunity.providers.weighted import WeightedScoreProvider
        assert cls is WeightedScoreProvider

    def test_get_provider_class_invalid(self) -> None:
        with pytest.raises(KeyError):
            get_scoring_provider_class("nonexistent")

    def test_create_provider(self) -> None:
        provider = create_scoring_provider("weighted")
        assert provider.name == "weighted"
        assert provider.priority == 100

    def test_sorted_order(self) -> None:
        names = sorted_scoring_providers()
        assert names.index("weighted") < names.index("competition")

    def test_register_custom_provider(self) -> None:
        @register_scoring_provider(name="custom_test", priority=50)
        class CustomProvider(ScoringProvider):
            @property
            def name(self) -> str:
                return "custom_test"
            @property
            def priority(self) -> int:
                return 50
            def score(self, candidate: dict, context: dict) -> ScoringBreakdown:
                return ScoringBreakdown()

        assert "custom_test" in available_scoring_providers()
        provider = create_scoring_provider("custom_test")
        assert provider.name == "custom_test"

    def test_empty_candidate(self) -> None:
        provider = create_scoring_provider("weighted")
        result = provider.score({}, {"max_evidence_count": 1, "total_platforms": 1, "max_product_count": 1})
        assert isinstance(result, ScoringBreakdown)
        assert 0.0 <= result.pain_severity <= 1.0
        assert 0.0 <= result.frequency <= 1.0


class TestWeightedScoreProvider:
    def test_score_basic(self) -> None:
        provider = create_scoring_provider("weighted")
        candidate = {
            "pain_severity": 0.8, "frequency_score": 0.7, "trend_score": 0.6,
            "evidence_count": 5, "reasoning_confidence": 0.75,
            "cluster_density": 0.6, "platform_count": 3,
            "product_count": 2, "competition_score": 0.4,
            "feasibility_score": 0.8, "novelty_score": 0.6,
        }
        context = {"max_evidence_count": 10, "total_platforms": 5, "max_product_count": 10}
        result = provider.score(candidate, context)
        assert result.pain_severity == 0.8
        assert result.frequency == 0.7
        assert result.evidence_count == 0.5

    def test_score_with_zero_evidence(self) -> None:
        provider = create_scoring_provider("weighted")
        candidate = {"pain_severity": 0.5, "evidence_count": 0}
        context = {"max_evidence_count": 1, "total_platforms": 1, "max_product_count": 1}
        result = provider.score(candidate, context)
        assert result.evidence_count <= 1.0


class TestMarketScoreProvider:
    def test_score_market(self) -> None:
        provider = create_scoring_provider("market")
        candidate = {"product_count": 5, "company_count": 3, "platform_count": 3, "evidence_count": 15}
        context = {"total_platforms": 5, "max_evidence_count": 20}
        result = provider.score(candidate, context)
        assert 0.0 <= result.market_coverage <= 1.0
        assert 0.0 <= result.feasibility <= 1.0


class TestTrendScoreProvider:
    def test_score_trend(self) -> None:
        provider = create_scoring_provider("trend")
        candidate = {"evidence_count": 10, "recent_evidence_ratio": 0.8, "evidence_growth_rate": 0.3}
        context = {}
        result = provider.score(candidate, context)
        assert 0.0 <= result.trend <= 1.0

    def test_score_trend_zero_evidence(self) -> None:
        provider = create_scoring_provider("trend")
        result = provider.score({}, {})
        assert 0.0 <= result.trend <= 1.0


class TestCompetitionScoreProvider:
    def test_score_competition(self) -> None:
        provider = create_scoring_provider("competition")
        candidate = {"product_count": 10, "company_count": 5}
        context = {"total_products": 50, "total_companies": 30}
        result = provider.score(candidate, context)
        assert 0.0 <= result.competition <= 1.0
        assert 0.0 <= result.novelty <= 1.0

    def test_no_competition(self) -> None:
        provider = create_scoring_provider("competition")
        candidate = {"product_count": 0, "company_count": 0}
        context = {"total_products": 10, "total_companies": 5}
        result = provider.score(candidate, context)
        assert result.competition >= 0.0


class TestBusinessModelProviders:
    def test_available(self) -> None:
        providers = available_business_model_providers()
        assert "saas" in providers
        assert "ai_agent" in providers
        assert "marketplace" in providers

    def test_saas_evaluate(self) -> None:
        provider = create_business_model_provider("saas")
        opp = Opportunity(
            opportunity_id="o1", title="Subscription Cloud Platform",
            summary="Monthly analytics dashboard", root_problem="p1",
        )
        model_type, score = provider.evaluate(opp, {})
        assert model_type == OpportunityType.SAAS
        assert 0.0 <= score <= 1.0

    def test_ai_agent_evaluate(self) -> None:
        provider = create_business_model_provider("ai_agent")
        opp = Opportunity(
            opportunity_id="o1",
            title="AI-powered chatbot",
            summary="Automate customer support with intelligent assistant",
            root_problem="p1",
        )
        model_type, score = provider.evaluate(opp, {})
        assert model_type == OpportunityType.AI_AGENT
        assert score > 0.0

    def test_no_keyword_match(self) -> None:
        provider = create_business_model_provider("saas")
        opp = Opportunity(
            opportunity_id="o1",
            title="Hardware device",
            summary="Physical product for consumers",
            root_problem="p1",
        )
        _, score = provider.evaluate(opp, {})
        assert score == 0.0

    def test_all_providers_valid(self) -> None:
        for name in available_business_model_providers():
            provider = create_business_model_provider(name)
            opp = Opportunity(
                opportunity_id="o1", title="Test", summary="Test", root_problem="p1",
            )
            model_type, score = provider.evaluate(opp, {})
            assert isinstance(model_type, OpportunityType)
            assert 0.0 <= score <= 1.0


class TestScoringProviderProtocol:
    def test_all_providers_return_breakdown(self) -> None:
        context = {"max_evidence_count": 10, "total_platforms": 5, "max_product_count": 10, "total_products": 50, "total_companies": 30}
        candidate = {
            "pain_severity": 0.8, "frequency_score": 0.7, "trend_score": 0.6,
            "evidence_count": 5, "reasoning_confidence": 0.75, "cluster_density": 0.6,
            "platform_count": 3, "product_count": 2, "company_count": 1,
            "competition_score": 0.4, "feasibility_score": 0.8, "novelty_score": 0.6,
            "recent_evidence_ratio": 0.8, "evidence_growth_rate": 0.3,
        }
        for name in available_scoring_providers():
            provider = create_scoring_provider(name)
            result = provider.score(candidate, context)
            assert isinstance(result, ScoringBreakdown)
            for field in ScoringBreakdown.model_fields:
                val = getattr(result, field)
                assert 0.0 <= val <= 1.0, f"{name}.{field} = {val}"
