from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InferenceResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    inference_id: str
    inference_type: str = ""
    derived_node_id: str = ""
    confidence: float = 0.0
    chain_id: str = ""


class ChainResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chain_id: str
    inference_id: str = ""
    total_confidence: float = 0.0
    steps_count: int = 0


class RootCauseResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cause_node_id: str = ""
    cause_label: str = ""
    effect_node_id: str = ""
    effect_label: str = ""
    path_length: int = 0
    propagated_confidence: float = 0.0
    ranking_score: float = 0.0


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conclusion_node_id: str = ""
    evidence_count: int = 0
    aggregated_confidence: float = 0.0
