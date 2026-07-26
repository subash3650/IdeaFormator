"""Tests for OpportunitySearch."""

from __future__ import annotations

from phase3.opportunity.schema import Opportunity
from phase3.opportunity.search import OpportunitySearch


class TestOpportunitySearch:
    def test_find_by_id(self) -> None:
        opps = [
            Opportunity(opportunity_id="o1", title="A", summary="S", root_problem="p"),
            Opportunity(opportunity_id="o2", title="B", summary="S", root_problem="p"),
        ]
        search = OpportunitySearch(opps)
        assert search.find_by_id("o1") is not None
        assert search.find_by_id("o3") is None

    def test_find_by_entity(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="A", summary="S", root_problem="p",
            affected_products=["ProductX", "ProductY"],
        )
        search = OpportunitySearch([opp])
        results = search.find_by_entity("ProductX")
        assert len(results) == 1
        results = search.find_by_entity("Nonexistent")
        assert len(results) == 0

    def test_find_by_company(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="A", summary="S", root_problem="p",
            affected_companies=["Acme Corp"],
        )
        search = OpportunitySearch([opp])
        results = search.find_by_company("Acme")
        assert len(results) == 1

    def test_find_by_product(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="A", summary="S", root_problem="p",
            affected_products=["Widget"],
        )
        search = OpportunitySearch([opp])
        results = search.find_by_product("Widget")
        assert len(results) == 1

    def test_find_by_cluster(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="A", summary="S", root_problem="p",
            cluster_ids=["cl1"],
        )
        search = OpportunitySearch([opp])
        results = search.find_by_cluster("cl1")
        assert len(results) == 1
        assert search.find_by_cluster("cl2") == []

    def test_find_by_reasoning_chain(self) -> None:
        opp = Opportunity(
            opportunity_id="o1", title="A", summary="S", root_problem="p",
            reasoning_chain_ids=["chain1"],
        )
        search = OpportunitySearch([opp])
        results = search.find_by_reasoning_chain("chain1")
        assert len(results) == 1

    def test_find_cross_platform(self) -> None:
        from phase3.opportunity.schema import ScoringBreakdown
        sb = ScoringBreakdown(cross_platform=0.7)
        opp = Opportunity(
            opportunity_id="o1", title="A", summary="S", root_problem="p",
            scoring_breakdown=sb,
        )
        search = OpportunitySearch([opp])
        results = search.find_cross_platform()
        assert len(results) == 1

    def test_find_emerging(self) -> None:
        from phase3.opportunity.schema import RecommendationType
        opp = Opportunity(
            opportunity_id="o1", title="A", summary="S", root_problem="p",
            trend_score=0.8,
            recommendation_type=RecommendationType.STRONG_PURSUE,
        )
        search = OpportunitySearch([opp])
        results = search.find_emerging(min_trend=0.6)
        assert len(results) == 1

    def test_search_text(self) -> None:
        opps = [
            Opportunity(opportunity_id="o1", title="Slow Performance",
                        summary="Users complain about speed", root_problem="p1"),
            Opportunity(opportunity_id="o2", title="Battery Drain",
                        summary="Battery issues on mobile", root_problem="p2"),
        ]
        search = OpportunitySearch(opps)
        results = search.search_text("performance", top_k=10)
        assert len(results) == 1
        assert results[0].opportunity_id == "o1"

    def test_search_text_no_match(self) -> None:
        opp = Opportunity(opportunity_id="o1", title="Test", summary="S", root_problem="p")
        search = OpportunitySearch([opp])
        results = search.search_text("nonexistent")
        assert results == []

    def test_search_text_empty_query(self) -> None:
        opp = Opportunity(opportunity_id="o1", title="Test", summary="S", root_problem="p")
        search = OpportunitySearch([opp])
        results = search.search_text("")
        assert len(results) == 0
