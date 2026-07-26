"""Tests for Pydantic models in the Reasoning Engine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from phase2.reasoning.schema import (
    ExplainabilityScore,
    Explanation,
    ExplanationFormat,
    InferenceResult,
    InferenceType,
    PropagationStrategy,
    ProvenanceVersion,
    ReasoningChain,
    ReasoningMetadata,
    ReasoningStep,
    RootCause,
    RootCauseRanking,
)


class TestInferenceType:
    def test_values(self) -> None:
        assert InferenceType.TRANSITIVE.value == "transitive"
        assert InferenceType.EVIDENCE_AGGREGATION.value == "evidence_aggregation"
        assert InferenceType.CAUSAL_CHAIN.value == "causal_chain"

    def test_unique(self) -> None:
        values = [t.value for t in InferenceType]
        assert len(values) == len(set(values))


class TestPropagationStrategy:
    def test_values(self) -> None:
        assert PropagationStrategy.MULTIPLICATIVE.value == "multiplicative"
        assert PropagationStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert PropagationStrategy.MINIMUM.value == "minimum"
        assert PropagationStrategy.DECAY.value == "decay"


class TestReasoningStep:
    def test_defaults(self) -> None:
        step = ReasoningStep(step_id=0, rule_name="test")
        assert step.step_id == 0
        assert step.rule_name == "test"
        assert step.input_node_ids == []
        assert step.input_edge_ids == []
        assert step.output_node_id is None
        assert step.output_edge_id is None
        assert step.confidence_delta == 0.0
        assert step.timestamp != ""

    def test_frozen(self) -> None:
        step = ReasoningStep(step_id=0, rule_name="test")
        with pytest.raises(ValidationError):
            ReasoningStep(step_id=-1, rule_name="test")

    def test_confidence_range(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningStep(step_id=0, rule_name="test", confidence_delta=1.5)


class TestInferenceResult:
    def test_minimal(self) -> None:
        result = InferenceResult(
            inference_id="abc123",
            inference_type=InferenceType.TRANSITIVE,
            chain_id="chain123",
            confidence=0.0,
        )
        assert result.inference_id == "abc123"
        assert result.inference_type == InferenceType.TRANSITIVE
        assert result.confidence == 0.0
        assert result.derived_node_id is None

    def test_confidence_range(self) -> None:
        with pytest.raises(ValidationError):
            InferenceResult(
                inference_id="x", inference_type=InferenceType.TRANSITIVE,
                chain_id="c", confidence=1.5,
            )


class TestReasoningChain:
    def test_defaults(self) -> None:
        chain = ReasoningChain(chain_id="c1", inference_id="i1")
        assert chain.chain_id == "c1"
        assert chain.inference_id == "i1"
        assert chain.steps == []
        assert chain.total_confidence == 0.0

    def test_provenance_version(self) -> None:
        pv = ProvenanceVersion(run_id="run123")
        assert pv.run_id == "run123"
        assert pv.rule_version == "1.0"

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningChain(chain_id="c1", inference_id="i1", unknown_field="x")


class TestRootCause:
    def test_minimal(self) -> None:
        rc = RootCause(
            cause_node_id="n1", cause_label="Cause",
            effect_node_id="n2", effect_label="Effect",
            path_length=2,
        )
        assert rc.cause_node_id == "n1"
        assert rc.ranking_method == RootCauseRanking.TRANSITIVE_IMPACT

    def test_element(self) -> None:
        rc = RootCause(
            cause_node_id="n1", cause_label="C",
            effect_node_id="n2", effect_label="E",
            path=["n3", "n1", "n2"], path_length=2,
            propagated_confidence=0.8, transitive_impact_count=5,
            evidence_count=3, ranking_score=0.9,
        )
        assert rc.path_length == 2
        assert rc.propagated_confidence == 0.8


class TestExplainabilityScore:
    def test_compute_high(self) -> None:
        result = ExplainabilityScore.compute(0.9, 5, 1)
        assert 0.5 < result.explanation_score <= 1.0
        assert result.confidence == 0.9
        assert result.evidence_count == 5

    def test_compute_low(self) -> None:
        result = ExplainabilityScore.compute(0.1, 0, 10)
        assert result.explanation_score < 0.3

    def test_compute_zero_evidence(self) -> None:
        result = ExplainabilityScore.compute(1.0, 0, 1)
        assert result.explanation_score < 1.0


class TestExplanation:
    def test_defaults(self) -> None:
        exp = Explanation(explanation_id="e1", inference_id="i1")
        assert exp.format == ExplanationFormat.TEMPLATE
        assert exp.title == ""


class TestReasoningMetadata:
    def test_defaults(self) -> None:
        meta = ReasoningMetadata(run_id="run1")
        assert meta.inference_count == 0
        assert meta.cache_hit is False


class TestProvenanceVersion:
    def test_defaults(self) -> None:
        pv = ProvenanceVersion()
        assert pv.rule_version == "1.0"
        assert pv.pipeline_version == "1.0"
        assert pv.run_id == ""


class TestSerialization:
    def test_inference_type_serialization(self) -> None:
        result = InferenceResult(
            inference_id="x", inference_type=InferenceType.CAUSAL_CHAIN, chain_id="c",
            confidence=0.5,
        )
        d = result.model_dump(mode="json")
        assert d["inference_type"] == "causal_chain"

    def test_no_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningStep(step_id=0, rule_name="t", extra="x")
