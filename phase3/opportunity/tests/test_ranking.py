"""Tests for OpportunityRanker."""

from __future__ import annotations

from pathlib import Path

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.ranking import OpportunityRanker
from phase3.opportunity.schema import (
    ConfidenceBreakdown,
    Opportunity,
    OpportunityStatus,
    RankingStrategy,
    ScoringBreakdown,
)


def _make_opp(oid: str, score: float, evidence: list[str] | None = None) -> Opportunity:
    return Opportunity(
        opportunity_id=oid,
        title=f"Opp {oid}",
        summary="S",
        root_problem=f"p_{oid}",
        opportunity_score=score,
        supporting_evidence=evidence or [f"ev_{oid}"],
        scoring_breakdown=ScoringBreakdown(),
        confidence=ConfidenceBreakdown(final_confidence=0.5),
        status=OpportunityStatus.SCORED,
    )


class TestOpportunityRanker:
    def test_rank_empty(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        ranker = OpportunityRanker(cfg)
        result = ranker.rank([])
        assert result == []

    def test_rank_sorts_by_score(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        ranker = OpportunityRanker(cfg)
        opps = [
            _make_opp("a", 0.5),
            _make_opp("b", 0.9),
            _make_opp("c", 0.3),
        ]
        result = ranker.rank(opps)
        assert len(result) == 3
        assert result[0].opportunity_id == "b"
        assert result[1].opportunity_id == "a"
        assert result[2].opportunity_id == "c"

    def test_rank_assigns_rank(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        ranker = OpportunityRanker(cfg)
        opps = [
            _make_opp("a", 0.5),
            _make_opp("b", 0.9),
        ]
        result = ranker.rank(opps)
        assert result[0].rank == 1
        assert result[1].rank == 2

    def test_rank_sets_status(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        ranker = OpportunityRanker(cfg)
        opps = [_make_opp("a", 0.5)]
        result = ranker.rank(opps)
        assert result[0].status == OpportunityStatus.RANKED

    def test_rank_respects_top_k(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, top_k=2)
        ranker = OpportunityRanker(cfg)
        opps = [_make_opp(f"o{i}", 0.1 * (i + 1)) for i in range(10)]
        result = ranker.rank(opps)
        assert len(result) == 2

    def test_rank_deduplicates(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, dedup_similarity_threshold=0.5)
        ranker = OpportunityRanker(cfg)
        opps = [
            _make_opp("a", 0.9, evidence=["ev1", "ev2", "ev3"]),
            _make_opp("b", 0.8, evidence=["ev1", "ev2", "ev3"]),
        ]
        result = ranker.rank(opps)
        assert len(result) == 1

    def test_rank_no_dedup_for_different_evidence(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, dedup_similarity_threshold=0.5)
        ranker = OpportunityRanker(cfg)
        opps = [
            _make_opp("a", 0.9, evidence=["ev1", "ev2"]),
            _make_opp("b", 0.8, evidence=["ev3", "ev4"]),
        ]
        result = ranker.rank(opps)
        assert len(result) == 2

    def test_rank_by_pain_severity(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, ranking_strategy=RankingStrategy.PAIN_SEVERITY)
        ranker = OpportunityRanker(cfg)
        opps = [
            Opportunity(
                opportunity_id="a", title="A", summary="S", root_problem="p",
                opportunity_score=0.5, pain_severity=0.3,
                scoring_breakdown=ScoringBreakdown(),
                confidence=ConfidenceBreakdown(final_confidence=0.5),
                status=OpportunityStatus.SCORED,
            ),
            Opportunity(
                opportunity_id="b", title="B", summary="S", root_problem="p",
                opportunity_score=0.5, pain_severity=0.9,
                scoring_breakdown=ScoringBreakdown(),
                confidence=ConfidenceBreakdown(final_confidence=0.5),
                status=OpportunityStatus.SCORED,
            ),
        ]
        result = ranker.rank(opps)
        assert result[0].opportunity_id == "b"

    def test_rank_by_confidence(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path, ranking_strategy=RankingStrategy.CONFIDENCE)
        ranker = OpportunityRanker(cfg)
        opps = [
            Opportunity(
                opportunity_id="a", title="A", summary="S", root_problem="p",
                opportunity_score=0.9,
                confidence=ConfidenceBreakdown(final_confidence=0.3),
                scoring_breakdown=ScoringBreakdown(),
                status=OpportunityStatus.SCORED,
            ),
            Opportunity(
                opportunity_id="b", title="B", summary="S", root_problem="p",
                opportunity_score=0.5,
                confidence=ConfidenceBreakdown(final_confidence=0.9),
                scoring_breakdown=ScoringBreakdown(),
                status=OpportunityStatus.SCORED,
            ),
        ]
        result = ranker.rank(opps)
        assert result[0].opportunity_id == "b"

    def test_evidence_overlap(self, tmp_path: Path) -> None:
        cfg = OpportunityConfig(output_dir=tmp_path)
        ranker = OpportunityRanker(cfg)
        overlap = ranker._evidence_overlap(
            _make_opp("a", 0.5, evidence=["a", "b", "c"]),
            _make_opp("b", 0.5, evidence=["a", "b", "d"]),
        )
        assert 0.5 <= overlap <= 1.0
