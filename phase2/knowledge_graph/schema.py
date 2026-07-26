"""Data models for the Knowledge Graph Infrastructure."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    DOCUMENT = "document"
    SOURCE = "source"
    OBSERVATION = "observation"
    ENTITY = "entity"
    KEYWORD = "keyword"
    PHRASE = "phrase"
    PATTERN = "pattern"
    CATEGORY = "category"
    PRODUCT = "product"
    COMPANY = "company"
    TECHNOLOGY = "technology"
    FEATURE = "feature"
    PROBLEM_SIGNAL = "problem_signal"
    EVIDENCE = "evidence"
    CLUSTER = "cluster"
    TOPIC = "topic"
    TREND = "trend"


class EdgeType(str, Enum):
    SIMILAR_TO = "similar_to"
    CO_OCCURS = "co_occurs"
    NEXT_TO = "next_to"
    DERIVED_FROM = "derived_from"
    REFERENCES = "references"
    MENTIONS = "mentions"
    BELONGS_TO = "belongs_to"
    CONTAINS = "contains"
    MEMBER_OF_CLUSTER = "member_of_cluster"
    HAS_ENTITY = "has_entity"
    HAS_KEYWORD = "has_keyword"
    HAS_PATTERN = "has_pattern"
    HAS_CATEGORY = "has_category"
    CAUSES = "causes"
    BLOCKS = "blocks"
    DEPENDS_ON = "depends_on"
    SUPPORTED_BY = "supported_by"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GraphNode(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    node_id: str = Field(description="Deterministic SHA-256 hash of node identity")
    node_type: NodeType = Field(description="Type of node")
    label: str = Field(description="Human-readable label, max 200 characters")
    properties: dict[str, Any] = Field(default_factory=dict, description="Domain-specific fields")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Provenance fields")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Computational fields")
    source_asset: str = Field(description="Source pipeline asset filename")
    source_id: str = Field(description="ID from originating pipeline asset")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score [0, 1]")
    created_at: str = Field(default_factory=_now_iso, description="ISO-8601 UTC timestamp")
    pipeline_version: str = Field(description="Pipeline version at creation")
    schema_version: str = Field(description="Schema version at creation")


class GraphEdge(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    edge_id: str = Field(description="Deterministic SHA-256 hash of edge identity")
    source_node_id: str = Field(description="Source node ID")
    target_node_id: str = Field(description="Target node ID")
    edge_type: EdgeType = Field(description="Type of relationship")
    weight: float = Field(ge=0.0, le=1.0, description="Relationship strength [0, 1]")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score [0, 1]")
    properties: dict[str, Any] = Field(default_factory=dict, description="Domain-specific fields")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Provenance fields")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Computational fields")
    source_asset: str = Field(description="Source pipeline asset filename")
    created_at: str = Field(default_factory=_now_iso, description="ISO-8601 UTC timestamp")
    pipeline_version: str = Field(description="Pipeline version at creation")
    schema_version: str = Field(description="Schema version at creation")


class GraphMetadata(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    graph_id: str = Field(description="Unique graph identifier")
    node_count: int = Field(default=0, ge=0, description="Total nodes")
    edge_count: int = Field(default=0, ge=0, description="Total edges")
    node_type_counts: dict[str, int] = Field(default_factory=dict, description="Node type distribution")
    edge_type_counts: dict[str, int] = Field(default_factory=dict, description="Edge type distribution")
    connected_components: int = Field(default=0, ge=0, description="Number of connected components")
    largest_component_size: int = Field(default=0, ge=0, description="Largest component node count")
    density: float = Field(default=0.0, ge=0.0, le=1.0, description="Graph density")
    avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Average confidence")
    avg_degree: float = Field(default=0.0, ge=0.0, description="Average node degree")
    orphan_node_count: int = Field(default=0, ge=0, description="Nodes with degree zero")
    created_at: str = Field(default_factory=_now_iso, description="ISO-8601 UTC timestamp")
    run_id: str = Field(description="Pipeline run ID")
    pipeline_version: str = Field(description="Pipeline version")
    schema_version: str = Field(description="Schema version")


class ValidationResult(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    valid: bool = Field(description="Overall validation result")
    errors: list[str] = Field(default_factory=list, description="Error messages (must fix)")
    warnings: list[str] = Field(default_factory=list, description="Warning messages (advisory)")
    node_count: int = Field(ge=0, default=0, description="Total nodes checked")
    edge_count: int = Field(ge=0, default=0, description="Total edges checked")
    duplicate_node_count: int = Field(ge=0, default=0, description="Duplicate node count")
    duplicate_edge_count: int = Field(ge=0, default=0, description="Duplicate edge count")
    orphan_node_count: int = Field(ge=0, default=0, description="Degree-zero nodes")
    orphan_edge_count: int = Field(ge=0, default=0, description="Edges referencing missing nodes")
    self_loop_count: int = Field(ge=0, default=0, description="Self-loop edges")
    cycle_count: int = Field(ge=0, default=0, description="Directed cycles detected")
    disconnected_components: int = Field(ge=0, default=0, description="Disconnected components")
    stale_asset_count: int = Field(ge=0, default=0, description="Stale assets detected")
    run_id_mismatch_count: int = Field(ge=0, default=0, description="Run ID mismatches")
    checksum_mismatch_count: int = Field(ge=0, default=0, description="Checksum mismatches")
    schema_mismatch_count: int = Field(ge=0, default=0, description="Schema mismatches")
