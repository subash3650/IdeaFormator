"""Pydantic models for the Knowledge Graph Reasoning Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InferenceType(str, Enum):
    TRANSITIVE = "transitive"
    EVIDENCE_AGGREGATION = "evidence_aggregation"
    CAUSAL_CHAIN = "causal_chain"


class PropagationStrategy(str, Enum):
    MULTIPLICATIVE = "multiplicative"
    WEIGHTED_AVERAGE = "weighted_average"
    MINIMUM = "minimum"
    DECAY = "decay"


class RootCauseRanking(str, Enum):
    TRANSITIVE_IMPACT = "transitive_impact"
    CONFIDENCE = "confidence"
    DEPTH = "depth"


class ExplanationFormat(str, Enum):
    STRUCTURED = "structured"
    TEMPLATE = "template"
    MARKDOWN = "markdown"


class ProvenanceVersion(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    rule_version: str = Field(default="1.0")
    pipeline_version: str = Field(default="1.0")
    graph_version: str = Field(default="1.0")
    run_id: str = Field(default="")


class ReasoningStep(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    step_id: int = Field(ge=0)
    rule_name: str
    rule_version: str = Field(default="1.0")
    input_node_ids: list[str] = Field(default_factory=list)
    input_edge_ids: list[str] = Field(default_factory=list)
    output_node_id: str | None = Field(default=None)
    output_edge_id: str | None = Field(default=None)
    confidence_delta: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: str = Field(default_factory=_now_iso)


class InferenceResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    inference_id: str
    inference_type: InferenceType
    derived_node_id: str | None = Field(default=None)
    derived_edge_id: str | None = Field(default=None)
    confidence: float = Field(ge=0.0, le=1.0)
    chain_id: str
    provenance: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)
    pipeline_version: str = Field(default="1.0")
    schema_version: str = Field(default="1.0")


class ReasoningChain(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    chain_id: str
    inference_id: str
    steps: list[ReasoningStep] = Field(default_factory=list)
    input_node_ids: list[str] = Field(default_factory=list)
    output_node_ids: list[str] = Field(default_factory=list)
    output_edge_ids: list[str] = Field(default_factory=list)
    total_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_version: ProvenanceVersion = Field(default_factory=ProvenanceVersion)
    created_at: str = Field(default_factory=_now_iso)


class RootCause(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    cause_node_id: str
    cause_label: str
    effect_node_id: str
    effect_label: str
    path: list[str] = Field(default_factory=list)
    path_length: int = Field(ge=0)
    propagated_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    transitive_impact_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    ranking_score: float = Field(default=0.0, ge=0.0)
    ranking_method: RootCauseRanking = RootCauseRanking.TRANSITIVE_IMPACT


class EvidenceAggregation(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    conclusion_node_id: str
    conclusion_label: str
    evidence_node_ids: list[str] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)
    aggregated_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    aggregation_method: str = ""
    conflicting_evidence_count: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=_now_iso)


class ExplainabilityScore(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(ge=0)
    reasoning_depth: int = Field(ge=0)
    explanation_score: float = Field(ge=0.0, le=1.0)

    @staticmethod
    def compute(confidence: float, evidence_count: int, reasoning_depth: int) -> ExplainabilityScore:
        evidence_factor = 1.0 - (0.5 ** (evidence_count / 3.0))
        depth_penalty = 1.0 / (reasoning_depth + 1.0) ** 0.3
        score = confidence * evidence_factor * depth_penalty
        return ExplainabilityScore(
            confidence=round(confidence, 4),
            evidence_count=evidence_count,
            reasoning_depth=reasoning_depth,
            explanation_score=round(max(0.0, min(1.0, score)), 4),
        )


class Explanation(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    explanation_id: str
    inference_id: str
    format: ExplanationFormat = ExplanationFormat.TEMPLATE
    title: str = ""
    summary: str = ""
    steps: list[str] = Field(default_factory=list)
    collapsed_step_count: int = Field(default=0, ge=0)
    evidence_summary: str = ""
    confidence_explanation: str = ""
    explainability_score: ExplainabilityScore | None = None
    raw_text: str = ""
    created_at: str = Field(default_factory=_now_iso)


class ReasoningMetadata(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    run_id: str
    kg_run_id: str = ""
    inference_count: int = Field(default=0, ge=0)
    chain_count: int = Field(default=0, ge=0)
    root_cause_count: int = Field(default=0, ge=0)
    explanation_count: int = Field(default=0, ge=0)
    rules_applied: list[str] = Field(default_factory=list)
    rule_firing_counts: dict[str, int] = Field(default_factory=dict)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    cache_hit: bool = Field(default=False)
    created_at: str = Field(default_factory=_now_iso)
    pipeline_version: str = Field(default="1.0")
    schema_version: str = Field(default="1.0")


class InferenceOutput(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    inferences: list[InferenceResult] = Field(default_factory=list)
    chains: list[ReasoningChain] = Field(default_factory=list)
    root_causes: list[RootCause] = Field(default_factory=list)
    evidence_aggregations: list[EvidenceAggregation] = Field(default_factory=list)
    explanations: list[Explanation] = Field(default_factory=list)
    metadata: ReasoningMetadata | None = None
