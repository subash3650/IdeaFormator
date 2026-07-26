"""Data models for the Semantic Clustering Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ClusterType(str, Enum):
    NORMAL = "normal"
    LOW_QUALITY = "low_quality"


class ClusterMember(BaseModel):
    """A single member within a semantic cluster."""

    member_id: str = Field(description="Unique identifier of the member concept")
    similarity_to_representative: float = Field(ge=0.0, le=1.0, description="Similarity score to cluster representative")
    degree: int = Field(ge=0, description="Number of relationships within the cluster")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional member metadata")

    model_config = {"frozen": True, "extra": "forbid"}


class SemanticCluster(BaseModel):
    """An immutable semantic cluster representing a customer problem community."""

    cluster_id: str = Field(description="Deterministic SHA-256 cluster identifier")
    representative_id: str = Field(description="ID of the representative concept")
    member_ids: tuple[str, ...] = Field(description="Sorted tuple of member IDs for deterministic ordering")
    member_count: int = Field(ge=0, description="Total number of members in cluster")
    relationship_count: int = Field(ge=0, description="Total internal relationships")
    average_similarity: float = Field(ge=0.0, le=1.0, description="Mean pairwise similarity")
    density: float = Field(ge=0.0, le=1.0, description="Cluster density (edges / possible edges)")
    quality_score: float = Field(ge=0.0, le=1.0, description="Composite quality score")
    cluster_type: ClusterType = Field(default=ClusterType.NORMAL, description="Quality classification")
    provider: str = Field(description="Clustering provider name")
    provider_version: str = Field(description="Provider version")
    algorithm: str = Field(description="Algorithm name (e.g., 'connected_components')")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional cluster metadata")
    version: str = Field(description="Schema/engine version")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO 8601 timestamp")

    model_config = {"frozen": True, "extra": "forbid"}


class ClusterManifest(BaseModel):
    """Manifest describing a cluster generation run."""

    project: str = "pain-intelligence-engine"
    phase: str = "2"
    module: str = "clustering"
    provider: str = Field(description="Clustering provider used")
    provider_version: str = Field(description="Provider version")
    algorithm: str = Field(description="Algorithm name")
    relationship_schema_version: str = "1.0"
    cluster_schema_version: str = "1.0"
    record_count: int = Field(description="Total clusters generated")
    member_count: int = Field(description="Total members across all clusters")
    relationship_count: int = Field(description="Total relationships used")
    generated_at: str = Field(description="ISO 8601 timestamp")
    elapsed_seconds: float = Field(description="Total generation time")
    config_hash: str = Field(description="SHA-256 of clustering configuration")
    relationship_manifest_hash: str = Field(description="SHA-256 of relationship manifest")
    checksums: dict[str, str] | None = Field(None, description="File checksums")

    model_config = {"extra": "forbid"}


class ClusterReport(BaseModel):
    """Quality report for generated clusters."""

    report_type: str = "cluster_report"
    generated_at: str = Field(description="ISO 8601 timestamp")
    run_id: str | None = Field(None, description="Optional run identifier")
    elapsed_seconds: float = Field(description="Total execution time")
    total_clusters: int = Field(description="Total clusters generated")
    total_members: int = Field(description="Total unique members")
    total_relationships: int = Field(description="Total relationships processed")
    cluster_size_distribution: dict[int, int] = Field(default_factory=dict, description="Size -> count")
    largest_clusters: list[dict[str, Any]] = Field(default_factory=list, description="Top largest clusters")
    smallest_clusters: list[dict[str, Any]] = Field(default_factory=list, description="Top smallest clusters")
    average_cluster_size: float = Field(description="Mean cluster size")
    cluster_density: float = Field(description="Overall density")
    quality_distribution: dict[str, int] = Field(default_factory=dict, description="Quality score bins")
    top_representative_ids: list[str] = Field(default_factory=list, description="Most central representatives")
    orphan_concept_count: int = Field(description="Concepts not in any cluster")
    singleton_count: int = Field(description="Clusters of size 1 (if retained)")
    low_quality_count: int = Field(description="Clusters below quality threshold")
    provider: str = Field(description="Clustering provider")
    algorithm: str = Field(description="Algorithm name")

    model_config = {"extra": "forbid"}


class ClusterSummary(BaseModel):
    """Lightweight summary for listing/search operations."""

    cluster_id: str
    representative_id: str
    member_count: int
    quality_score: float
    cluster_type: ClusterType

    model_config = {"frozen": True, "extra": "forbid"}


class ClusterMetrics(BaseModel):
    """Detailed metrics for a single cluster."""

    cluster_id: str
    member_count: int
    relationship_count: int
    average_similarity: float
    density: float
    average_degree: float
    max_degree: int
    min_degree: int
    edge_count: int
    internal_edge_count: int
    external_edge_count: int
    internal_cohesion: float
    external_separation: float
    connectivity: float
    quality_score: float

    model_config = {"frozen": True, "extra": "forbid"}


class ClusterStats(BaseModel):
    """Aggregate statistics across all clusters."""

    total_clusters: int
    total_members: int
    total_relationships: int
    average_cluster_size: float
    average_density: float
    average_quality: float
    cluster_size_min: int
    cluster_size_max: int
    cluster_size_median: float
    quality_distribution: dict[str, int]
    cluster_type_counts: dict[str, int]

    model_config = {"extra": "forbid"}


class ValidationIssue(BaseModel):
    """A single validation diagnostic."""

    severity: str = Field(description="ERROR, WARN, or INFO")
    code: str = Field(description="Machine-readable issue code")
    message: str = Field(description="Human-readable description")
    cluster_id: str | None = Field(None, description="Affected cluster, if applicable")
    member_id: str | None = Field(None, description="Affected member, if applicable")

    model_config = {"frozen": True, "extra": "forbid"}


class ValidationResult(BaseModel):
    """Result of cluster validation."""

    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    clusters_checked: int = 0
    members_checked: int = 0

    model_config = {"extra": "forbid"}