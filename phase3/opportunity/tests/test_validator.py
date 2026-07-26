"""Tests for OpportunityValidator."""

from __future__ import annotations

from phase3.opportunity.schema import (
    ConfidenceBreakdown,
    Opportunity,
    OpportunityMetadata,
    ScoringBreakdown,
)
from phase3.opportunity.validator import OpportunityValidator


class TestOpportunityValidator:
    def test_valid_empty(self) -> None:
        validator = OpportunityValidator()
        result = validator.validate([])
        assert result.valid is True
        assert "No opportunities to validate" in result.warnings

    def test_valid_opportunities(self) -> None:
        validator = OpportunityValidator()
        opps = [
            Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p"),
            Opportunity(opportunity_id="o2", title="T2", summary="S2", root_problem="p2"),
        ]
        result = validator.validate(opps)
        assert result.valid is True
        assert result.opportunities_checked == 2

    def test_duplicate_ids(self) -> None:
        validator = OpportunityValidator()
        opps = [
            Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p"),
            Opportunity(opportunity_id="o1", title="T2", summary="S2", root_problem="p2"),
        ]
        result = validator.validate(opps)
        assert result.valid is False
        assert result.duplicate_count > 0
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_missing_evidence_warning(self) -> None:
        validator = OpportunityValidator()
        opps = [
            Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p",
                        supporting_evidence=[]),
        ]
        result = validator.validate(opps)
        assert result.missing_evidence_count == 1

    def test_broken_chain_references(self) -> None:
        validator = OpportunityValidator()
        opps = [
            Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p",
                        reasoning_chain_ids=["nonexistent_chain"],
                        supporting_evidence=["ev1"]),
        ]
        result = validator.validate(opps, valid_chain_ids={"real_chain"})
        assert result.broken_reference_count == 1

    def test_invalid_scores(self) -> None:
        validator = OpportunityValidator()
        opp = Opportunity(
            opportunity_id="o1", title="T", summary="S", root_problem="p",
            opportunity_score=0.5,
            supporting_evidence=["ev1"],
            confidence=ConfidenceBreakdown(final_confidence=0.5),
        )
        # Score is valid by default
        assert 0.0 <= opp.opportunity_score <= 1.0
        result = validator.validate([opp])
        assert result.invalid_score_count == 0

    def test_missing_confidence_warning(self) -> None:
        validator = OpportunityValidator()
        opp = Opportunity(
            opportunity_id="o1", title="T", summary="S", root_problem="p",
            opportunity_score=0.8,
        )
        # By default confidence.final_confidence == 0.0, so this should warn
        result = validator.validate([opp])
        assert result.missing_confidence_count == 1

    def test_manifest_consistency(self) -> None:
        validator = OpportunityValidator()
        opps = [Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p")]
        meta = OpportunityMetadata(run_id="r1", total_opportunities=5)
        result = validator.validate(opps, metadata=meta)
        assert any("manifest" in w.lower() for w in result.warnings)

    def test_all_valid(self) -> None:
        validator = OpportunityValidator()
        opps = [
            Opportunity(
                opportunity_id="o1", title="T", summary="S", root_problem="p",
                supporting_evidence=["ev1"], reasoning_chain_ids=["c1"],
                opportunity_score=0.8,
                confidence=ConfidenceBreakdown(final_confidence=0.8),
            ),
        ]
        result = validator.validate(opps, valid_chain_ids={"c1"})
        assert result.valid is True
        assert result.errors == []
