"""Tests for deterministic behavior of the Opportunity Engine."""

from __future__ import annotations

from pathlib import Path

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.schema import Opportunity
from phase3.opportunity.store import OpportunityStore


class TestDeterminism:
    def test_opportunity_id_deterministic(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        opps1 = [
            Opportunity(opportunity_id="o1", title="Test Opp", summary="S1", root_problem="p1"),
            Opportunity(opportunity_id="o2", title="Test Opp 2", summary="S2", root_problem="p2"),
        ]
        opps2 = [
            Opportunity(opportunity_id="o1", title="Test Opp", summary="S1", root_problem="p1"),
            Opportunity(opportunity_id="o2", title="Test Opp 2", summary="S2", root_problem="p2"),
        ]
        store.save_opportunities(opps1, "run1")
        store.save_opportunities(opps2, "run2")
        loaded1 = store.load_opportunities()
        loaded2 = store.load_opportunities()
        for o1, o2 in zip(loaded1, loaded2):
            assert o1.opportunity_id == o2.opportunity_id
            assert o1.title == o2.title

    def test_checksums_deterministic(self, tmp_path: Path) -> None:
        store1 = OpportunityStore(tmp_path / "a")
        store2 = OpportunityStore(tmp_path / "b")
        opps = [
            Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p"),
        ]
        store1.save_opportunities(opps, "run1")
        store2.save_opportunities(opps, "run1")
        assert store1.checksums() == store2.checksums()

    def test_store_load_does_not_mutate(self, tmp_path: Path) -> None:
        store = OpportunityStore(tmp_path)
        opp = Opportunity(opportunity_id="o1", title="T", summary="S", root_problem="p")
        original_id = opp.opportunity_id
        store.save_opportunities([opp], "run1")
        loaded = store.load_opportunities()
        assert loaded[0].opportunity_id == original_id
        assert loaded[0].title == "T"
