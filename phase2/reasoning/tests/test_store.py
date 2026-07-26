"""Tests for ReasoningStore — Parquet persistence round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase2.reasoning.schema import (
    EvidenceAggregation,
    ExplainabilityScore,
    InferenceResult,
    InferenceType,
    ReasoningChain,
    ReasoningMetadata,
    ReasoningStep,
    RootCause,
    RootCauseRanking,
)
from phase2.reasoning.store import ReasoningStore


class TestReasoningStore:
    def test_save_and_load_inferences(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        inferences = [
            InferenceResult(
                inference_id="inf1",
                inference_type=InferenceType.TRANSITIVE,
                chain_id="chain1",
                confidence=0.85,
                provenance=["a", "b", "c"],
            ),
            InferenceResult(
                inference_id="inf2",
                inference_type=InferenceType.CAUSAL_CHAIN,
                chain_id="chain2",
                confidence=0.72,
                provenance=["x", "y"],
            ),
        ]
        store.save_inferences(inferences, "test-run")
        loaded = store.load_inferences()
        assert len(loaded) == 2
        assert loaded[0].inference_id == "inf1"
        assert loaded[0].inference_type == InferenceType.TRANSITIVE
        assert loaded[1].confidence == 0.72

    def test_save_and_load_empty_inferences(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        store.save_inferences([], "test-run")
        loaded = store.load_inferences()
        assert loaded == []

    def test_save_and_load_chains(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        step = ReasoningStep(step_id=0, rule_name="test", output_node_id="n3", confidence_delta=0.8)
        chains = [
            ReasoningChain(
                chain_id="chain1",
                inference_id="inf1",
                steps=[step],
                input_node_ids=["n1", "n2"],
                output_node_ids=["n3"],
                total_confidence=0.8,
            ),
        ]
        store.save_chains(chains, "test-run")
        loaded = store.load_chains()
        assert len(loaded) == 1
        assert loaded[0].chain_id == "chain1"
        assert len(loaded[0].steps) == 1
        assert loaded[0].steps[0].rule_name == "test"

    def test_save_and_load_root_causes(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        causes = [
            RootCause(
                cause_node_id="c1",
                cause_label="Cause 1",
                effect_node_id="e1",
                effect_label="Effect 1",
                path=["c1", "m1", "e1"],
                path_length=2,
                propagated_confidence=0.8,
                transitive_impact_count=3,
                evidence_count=2,
                ranking_score=0.9,
            ),
        ]
        store.save_root_causes(causes, "test-run")
        loaded = store.load_root_causes()
        assert len(loaded) == 1
        assert loaded[0].cause_node_id == "c1"
        assert loaded[0].ranking_method == RootCauseRanking.TRANSITIVE_IMPACT

    def test_save_and_load_evidence(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        evs = [
            EvidenceAggregation(
                conclusion_node_id="target",
                conclusion_label="Target",
                evidence_node_ids=["e1", "e2"],
                evidence_count=2,
                aggregated_confidence=0.85,
                aggregation_method="weighted_average",
            ),
        ]
        store.save_evidence_aggregations(evs, "test-run")
        loaded = store.load_evidence_aggregations()
        assert len(loaded) == 1
        assert loaded[0].conclusion_node_id == "target"
        assert loaded[0].evidence_count == 2

    def test_save_metadata(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        meta = ReasoningMetadata(
            run_id="run1",
            inference_count=10,
            chain_count=5,
        )
        store.save_metadata(meta)
        loaded = store.load_metadata()
        assert loaded is not None
        assert loaded.run_id == "run1"
        assert loaded.inference_count == 10

    def test_load_metadata_nonexistent(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        loaded = store.load_metadata()
        assert loaded is None

    def test_file_structure(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        assert store.reasoning_dir == tmp_path / "reasoning"
        assert store.reasoning_dir.exists()

    def test_checksums(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        store.save_inferences([
            InferenceResult(inference_id="x", inference_type=InferenceType.TRANSITIVE, chain_id="c", confidence=0.5)
        ], "run1")
        cs = store.checksums()
        assert "derived_inferences.parquet" in cs
        assert len(cs["derived_inferences.parquet"]) == 16

    def test_checksums_empty(self, tmp_path: Path) -> None:
        store = ReasoningStore(tmp_path)
        cs = store.checksums()
        assert cs == {}
