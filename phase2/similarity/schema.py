"""Data models for the Semantic Relationship Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from phase2.embeddings.schema import SourceType


class RelationshipType(str, Enum):
    SIMILAR = "similar"
    DUPLICATE = "duplicate"
    ALTERNATIVE = "alternative"
    PARENT = "parent"
    CAUSES = "causes"
    SUBPROBLEM = "subproblem"


class SemanticRelationship(BaseModel):
    """A single semantic relationship between two knowledge items."""

    relationship_id: str = Field(description="SHA-256 of source_id|target_id|metric|version")
    source_type: SourceType = Field(description="Source asset type")
    source_id: str = Field(description="Source record ID")
    target_type: SourceType = Field(description="Target asset type")
    target_id: str = Field(description="Target record ID")
    relationship_type: RelationshipType = Field(
        default=RelationshipType.SIMILAR,
        description="Relationship classification",
    )
    similarity_score: float = Field(ge=0.0, le=1.0, description="Raw similarity score")
    confidence: float = Field(ge=0.0, le=1.0, description="Computed confidence score")
    metric: str = Field(description="Similarity metric used (cosine, dot_product, euclidean)")
    provider: str = Field(description="Provider name (cosine, dot_product, euclidean)")
    model_fingerprint: str = Field(description="Deterministic fingerprint: provider/model@dimension")
    shared_entities: list[str] = Field(default_factory=list, description="Shared entity identifiers")
    shared_categories: list[str] = Field(default_factory=list, description="Shared category tags")
    support_count: int = Field(default=0, description="Number of supporting evidence items")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(description="Schema/engine version")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp",
    )

    model_config = {"frozen": True, "extra": "forbid"}


class RelationshipJob(BaseModel):
    """Configuration for a single source-type processing job."""

    source_type: SourceType = Field(description="Source type to process")
    allowed_target_types: list[SourceType] = Field(description="Target types to compare against")
    metric: str = Field(description="Similarity metric")
    threshold: float = Field(ge=0.0, le=1.0, description="Minimum similarity threshold")
    top_k: int = Field(ge=1, description="Maximum neighbors per source")
    version: str = Field(description="Schema version")


class RelationshipManifest(BaseModel):
    """Manifest describing a relationship generation run."""

    project: str = "pain-intelligence-engine"
    phase: str = "2"
    module: str = "relationships"
    embedding_model: str = Field(description="Embedding model name")
    embedding_fingerprint: str = Field(description="Embedding model fingerprint")
    metric: str = Field(description="Similarity metric used")
    threshold: float = Field(description="Similarity threshold")
    relationship_schema_version: str = "1.0"
    record_count: int = Field(description="Total relationships generated")
    source_counts: dict[str, int] = Field(default_factory=dict, description="Count by source type")
    target_counts: dict[str, int] = Field(default_factory=dict, description="Count by target type")
    generated_at: str = Field(description="ISO 8601 timestamp")
    elapsed_seconds: float = Field(description="Total generation time")
    checksums: dict[str, str] | None = Field(None, description="File checksums")

    model_config = {"extra": "forbid"}
