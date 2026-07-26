from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NodeResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    node_type: str = ""
    label: str = ""
    confidence: float = 0.0
    properties: dict[str, Any] = Field(default_factory=dict)


class EdgeResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str = ""
    weight: float = 0.0
    confidence: float = 0.0


class GraphStatsResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node_count: int = 0
    edge_count: int = 0
    node_types: dict[str, int] = Field(default_factory=dict)
    edge_types: dict[str, int] = Field(default_factory=dict)
    density: float = 0.0
    avg_confidence: float = 0.0
