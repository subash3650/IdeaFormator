"""Tests for ChainTracker."""

from __future__ import annotations

import pytest

from phase2.reasoning.chains import ChainTracker
from phase2.reasoning.schema import ProvenanceVersion


class TestChainTracker:
    def test_start_chain(self) -> None:
        ct = ChainTracker(run_id="test-run")
        chain_id = ct.start_chain(inference_id="inf-1")
        assert chain_id != ""
        assert ct.chain_count() == 1

    def test_start_chain_with_inputs(self) -> None:
        ct = ChainTracker()
        chain_id = ct.start_chain(
            inference_id="inf-1",
            input_node_ids=["n1", "n2"],
            provenance_version=ProvenanceVersion(run_id="run-1"),
        )
        chain = ct.get_chain(chain_id)
        assert chain is not None
        assert chain.input_node_ids == ["n1", "n2"]
        assert chain.provenance_version.run_id == "run-1"

    def test_add_step(self) -> None:
        ct = ChainTracker()
        chain_id = ct.start_chain("inf-1")
        step = ct.add_step(
            chain_id=chain_id,
            rule_name="transitive_closure",
            input_node_ids=["n1", "n2"],
            output_node_id="n3",
            confidence_delta=0.85,
        )
        assert step.step_id == 0
        assert step.rule_name == "transitive_closure"
        assert step.output_node_id == "n3"

    def test_add_multiple_steps(self) -> None:
        ct = ChainTracker()
        chain_id = ct.start_chain("inf-1")
        ct.add_step(chain_id=chain_id, rule_name="rule1", confidence_delta=0.5)
        ct.add_step(chain_id=chain_id, rule_name="rule2", confidence_delta=0.3)
        chain = ct.finalize(chain_id, total_confidence=0.4)
        assert len(chain.steps) == 2

    def test_finalize(self) -> None:
        ct = ChainTracker()
        chain_id = ct.start_chain("inf-1")
        ct.add_step(chain_id=chain_id, rule_name="rule1", confidence_delta=0.8)
        chain = ct.finalize(
            chain_id=chain_id,
            total_confidence=0.8,
            output_node_ids=["n3"],
        )
        assert chain.total_confidence == 0.8
        assert chain.output_node_ids == ["n3"]

    def test_get_chains_for_inference(self) -> None:
        ct = ChainTracker()
        id1 = ct.start_chain("inf-1")
        id2 = ct.start_chain("inf-1")
        id3 = ct.start_chain("inf-2")
        chains = ct.get_chains_for_inference("inf-1")
        assert len(chains) == 2

    def test_all_chains(self) -> None:
        ct = ChainTracker()
        ct.start_chain("inf-1")
        ct.start_chain("inf-2")
        assert len(ct.all_chains()) == 2

    def test_unknown_chain(self) -> None:
        ct = ChainTracker()
        with pytest.raises(ValueError):
            ct.finalize("nonexistent", total_confidence=0.5)
