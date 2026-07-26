"""Tests for Pydantic models in the Opportunity Engine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from phase3.opportunity.schema import (
    ConfidenceBreakdown,
    ImplementationComplexity,
    MarketMaturity,
    MarketSize,
    Opportunity,
    OpportunityMetadata,
    OpportunityOutput,
    OpportunityStatus,
    OpportunityType,
    RecommendationType,
    ScoreWeights,
    ScoringBreakdown,
)


class TestEnums:
    def test_opportunity_type_values(self) -> None:
        assert OpportunityType.SAAS.value == "saas"
        assert OpportunityType.AI_AGENT.value == "ai_agent"
        assert OpportunityType.MARKETPLACE.value == "marketplace"

    def test_opportunity_type_unique(self) -> None:
        values = [t.value for t in OpportunityType]
        assert len(values) == len(set(values))

    def test_recommendation_type_values(self) -> None:
        assert RecommendationType.STRONG_PURSUE.value == "strong_pursue"
        assert RecommendationType.INSUFFICIENT_DATA.value == "insufficient_data"

    def test_opportunity_status_lifecycle(self) -> None:
        assert OpportunityStatus.IDENTIFIED.value == "identified"
        assert OpportunityStatus.VALIDATED.value == "validated"
        assert OpportunityStatus.SCORED.value == "scored"
        assert OpportunityStatus.RANKED.value == "ranked"
        assert OpportunityStatus.RECOMMENDED.value == "recommended"
        assert OpportunityStatus.PUBLISHED.value == "published"
        assert OpportunityStatus.ARCHIVED.value == "archived"

    def test_ranking_strategy_values(self) -> None:
        from phase3.opportunity.schema import RankingStrategy
        assert RankingStrategy.COMPOSITE.value == "composite"

    def test_market_enums(self) -> None:
        assert MarketSize.LARGE.value == "large"
        assert MarketMaturity.EMERGING.value == "emerging"
        assert ImplementationComplexity.LOW.value == "low"


class TestScoreWeights:
    def test_defaults(self) -> None:
        w = ScoreWeights()
        assert w.pain_severity == 0.20
        assert w.frequency == 0.15
        assert w.trend == 0.10
        assert w.evidence_count == 0.10
        assert w.cluster_density == 0.08
        assert w.cross_platform == 0.07
        assert w.market_coverage == 0.05
        assert w.competition == 0.05
        assert w.feasibility == 0.05
        assert w.novelty == 0.05

    def test_frozen(self) -> None:
        w = ScoreWeights()
        with pytest.raises(ValidationError):
            w.pain_severity = 0.5

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            ScoreWeights(unknown_field=1.0)

    def test_normalize(self) -> None:
        w = ScoreWeights(pain_severity=0.5, frequency=0.5)
        n = w.normalize()
        total = n.pain_severity + n.frequency + n.trend + n.evidence_count
        total += n.reasoning_confidence + n.cluster_density + n.cross_platform
        total += n.market_coverage + n.competition + n.feasibility + n.novelty
        assert abs(total - 1.0) < 1e-6

    def test_normalize_zero(self) -> None:
        w = ScoreWeights(pain_severity=0.0, frequency=0.0, trend=0.0,
                         evidence_count=0.0, reasoning_confidence=0.0,
                         cluster_density=0.0, cross_platform=0.0,
                         market_coverage=0.0, competition=0.0,
                         feasibility=0.0, novelty=0.0)
        n = w.normalize()
        assert n.pain_severity == 0.20

    def test_range_validation(self) -> None:
        with pytest.raises(ValidationError):
            ScoreWeights(pain_severity=1.5)

    def test_field_constraints(self) -> None:
        for field_name in ScoreWeights.model_fields:
            with pytest.raises(ValidationError):
                ScoreWeights(**{field_name: 1.5})


class TestScoringBreakdown:
    def test_defaults(self) -> None:
        sb = ScoringBreakdown()
        assert sb.pain_severity == 0.0
        assert sb.frequency == 0.0
        assert sb.trend == 0.0

    def test_frozen(self) -> None:
        sb = ScoringBreakdown()
        with pytest.raises(ValidationError):
            sb.pain_severity = 0.5

    def test_range(self) -> None:
        with pytest.raises(ValidationError):
            ScoringBreakdown(pain_severity=-0.1)
        with pytest.raises(ValidationError):
            ScoringBreakdown(pain_severity=1.5)


class TestConfidenceBreakdown:
    def test_defaults(self) -> None:
        cb = ConfidenceBreakdown()
        assert cb.final_confidence == 0.0
        assert cb.computation_method == "weighted_average"

    def test_frozen(self) -> None:
        cb = ConfidenceBreakdown()
        with pytest.raises(ValidationError):
            cb.final_confidence = 1.0

    def test_range(self) -> None:
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(reasoning_confidence=1.5)


class TestOpportunity:
    def test_minimal(self) -> None:
        opp = Opportunity(
            opportunity_id="opp123",
            title="Test Opportunity",
            summary="A test",
            root_problem="problem_1",
        )
        assert opp.opportunity_id == "opp123"
        assert opp.title == "Test Opportunity"
        assert opp.root_problem == "problem_1"
        assert opp.opportunity_score == 0.0
        assert opp.status == OpportunityStatus.IDENTIFIED
        assert opp.recommendation_type == RecommendationType.INSUFFICIENT_DATA
        assert opp.suggested_business_model == OpportunityType.SAAS
        assert opp.estimated_market_size == MarketSize.UNKNOWN

    def test_frozen(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="T", summary="S", root_problem="p",
        )
        with pytest.raises(ValidationError):
            opp.title = "New Title"

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            Opportunity(
                opportunity_id="o1", title="T", summary="S", root_problem="p",
                unknown_field="x",
            )

    def test_full(self) -> None:
        sb = ScoringBreakdown(pain_severity=0.8, frequency=0.7)
        cb = ConfidenceBreakdown(final_confidence=0.85)
        opp = Opportunity(
            opportunity_id="opp_full",
            title="Full Opportunity",
            summary="Full description",
            root_problem="root_1",
            supporting_evidence=["ev1", "ev2"],
            reasoning_chain_ids=["chain1"],
            cluster_ids=["cl1"],
            kg_node_ids=["n1", "n2"],
            affected_products=["ProductA"],
            affected_companies=["CompanyB"],
            affected_technologies=["TechC"],
            estimated_market_size=MarketSize.LARGE,
            pain_severity=0.9,
            frequency_score=0.8,
            trend_score=0.7,
            competition_score=0.3,
            feasibility_score=0.8,
            opportunity_score=0.85,
            scoring_breakdown=sb,
            confidence=cb,
            recommendation_type=RecommendationType.STRONG_PURSUE,
            suggested_solution="Build a SaaS",
            suggested_business_model=OpportunityType.SAAS,
            status=OpportunityStatus.RECOMMENDED,
            rank=1,
        )
        assert opp.estimated_market_size == MarketSize.LARGE
        assert opp.opportunity_score == 0.85
        assert opp.scoring_breakdown.pain_severity == 0.8
        assert opp.confidence.final_confidence == 0.85
        assert len(opp.affected_products) == 1
        assert opp.rank == 1

    def test_confidence_defaults_to_zero(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="T", summary="S", root_problem="p",
        )
        assert opp.confidence.final_confidence == 0.0
        assert opp.confidence.reasoning_confidence == 0.0

    def test_scoring_breakdown_defaults(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="T", summary="S", root_problem="p",
        )
        assert opp.scoring_breakdown.pain_severity == 0.0
        assert opp.scoring_breakdown.frequency == 0.0

    def test_string_representation(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="My Opp", summary="S", root_problem="p",
        )
        assert "My Opp" in str(opp.model_dump())

    def test_serialization_roundtrip(self) -> None:
        sb = ScoringBreakdown(pain_severity=0.6, frequency=0.5)
        opp = Opportunity(
            opportunity_id="o1", title="T", summary="S", root_problem="p",
            scoring_breakdown=sb,
        )
        data = opp.model_dump(mode="json")
        restored = Opportunity(**data)
        assert restored.opportunity_id == opp.opportunity_id
        assert restored.scoring_breakdown.pain_severity == 0.6

    def test_status_default(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="T", summary="S", root_problem="p",
        )
        assert opp.status == OpportunityStatus.IDENTIFIED

    def test_rank_default(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="T", summary="S", root_problem="p",
        )
        assert opp.rank == 0


class TestOpportunityMetadata:
    def test_defaults(self) -> None:
        meta = OpportunityMetadata(run_id="run1")
        assert meta.run_id == "run1"
        assert meta.total_opportunities == 0
        assert meta.cache_hit is False

    def test_frozen(self) -> None:
        meta = OpportunityMetadata(run_id="run1")
        with pytest.raises(ValidationError):
            meta.total_opportunities = 10


class TestOpportunityOutput:
    def test_defaults(self) -> None:
        output = OpportunityOutput()
        assert output.opportunities == []
        assert output.metadata is None

    def test_with_data(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="T", summary="S", root_problem="p",
        )
        meta = OpportunityMetadata(run_id="run1")
        output = OpportunityOutput(opportunities=[opp], metadata=meta)
        assert len(output.opportunities) == 1
        assert output.metadata.run_id == "run1"
