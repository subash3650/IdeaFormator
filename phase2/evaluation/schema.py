from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageHealth:
    score: float = 0.0
    max_score: float = 100.0
    weight: float = 1.0
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    degradation_pct: float = 0.0


@dataclass
class ReasoningEvaluation:
    inference_count: int = 0
    chain_count: int = 0
    root_cause_count: int = 0
    evidence_aggregation_count: int = 0
    rules_applied: list[str] = field(default_factory=list)
    avg_inference_confidence: float = 0.0
    avg_root_cause_depth: float = 0.0
    has_reasoning: bool = False
    health: StageHealth = field(default_factory=StageHealth)


@dataclass
class HistogramBin:
    label: str
    count: int
    percentage: float = 0.0


@dataclass
class DistributionStats:
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    histogram: list[HistogramBin] = field(default_factory=list)


@dataclass
class DocumentEvaluation:
    total_documents: int = 0
    documents_per_source: dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    duplicate_rate: float = 0.0
    avg_document_length: float = 0.0
    avg_document_length_chars: float = 0.0
    language_distribution: dict[str, int] = field(default_factory=dict)
    missing_fields: dict[str, int] = field(default_factory=dict)
    empty_content_count: int = 0
    empty_content_rate: float = 0.0
    metadata_completeness: float = 0.0
    collection_freshness_days: float = 0.0
    date_range: dict[str, str] = field(default_factory=dict)
    health: StageHealth = field(default_factory=StageHealth)


@dataclass
class ObservationEvaluation:
    total_observations: int = 0
    total_documents: int = 0
    observations_per_document: DistributionStats = field(default_factory=DistributionStats)
    entity_precision: float = 0.0
    entity_coverage: float = 0.0
    keyword_diversity: float = 0.0
    phrase_diversity: float = 0.0
    pattern_diversity: float = 0.0
    extractor_contribution: dict[str, int] = field(default_factory=dict)
    extractor_contribution_pct: dict[str, float] = field(default_factory=dict)
    canonicalization_success_rate: float = 0.0
    knowledge_enrichment_coverage: float = 0.0
    type_distribution: dict[str, int] = field(default_factory=dict)
    health: StageHealth = field(default_factory=StageHealth)


@dataclass
class EvidenceEvaluation:
    total_evidence: int = 0
    total_observations: int = 0
    compression_ratio: float = 0.0
    support_distribution: DistributionStats = field(default_factory=DistributionStats)
    category_distribution: dict[str, int] = field(default_factory=dict)
    evidence_confidence: DistributionStats = field(default_factory=DistributionStats)
    avg_observations_per_evidence: float = 0.0
    entity_type_distribution: dict[str, int] = field(default_factory=dict)
    health: StageHealth = field(default_factory=StageHealth)


@dataclass
class SignalEvaluation:
    total_candidates: int = 0
    accepted_signals: int = 0
    filtered_signals: int = 0
    filter_reasons: dict[str, int] = field(default_factory=dict)
    support_distribution: DistributionStats = field(default_factory=DistributionStats)
    confidence_distribution: DistributionStats = field(default_factory=DistributionStats)
    category_coverage: dict[str, int] = field(default_factory=dict)
    category_coverage_pct: float = 0.0
    zero_signal_explanation: str = ""
    health: StageHealth = field(default_factory=StageHealth)


@dataclass
class EmbeddingEvaluation:
    total_vectors: int = 0
    dimension: int = 0
    provider: str = ""
    model: str = ""
    duplicate_vector_count: int = 0
    duplicate_vector_rate: float = 0.0
    zero_vector_count: int = 0
    zero_vector_rate: float = 0.0
    vector_norm_distribution: DistributionStats = field(default_factory=DistributionStats)
    per_source_counts: dict[str, int] = field(default_factory=dict)
    health: StageHealth = field(default_factory=StageHealth)


@dataclass
class RelationshipEvaluation:
    total_relationships: int = 0
    average_similarity: float = 0.0
    similarity_distribution: DistributionStats = field(default_factory=DistributionStats)
    relationship_type_distribution: dict[str, int] = field(default_factory=dict)
    degree_distribution: DistributionStats = field(default_factory=DistributionStats)
    isolated_nodes: int = 0
    isolated_node_rate: float = 0.0
    largest_connected_component_size: int = 0
    largest_connected_component_pct: float = 0.0
    confidence_distribution: DistributionStats = field(default_factory=DistributionStats)
    health: StageHealth = field(default_factory=StageHealth)


@dataclass
class ClusterEvaluation:
    total_clusters: int = 0
    total_members: int = 0
    cluster_size_distribution: DistributionStats = field(default_factory=DistributionStats)
    quality_distribution: DistributionStats = field(default_factory=DistributionStats)
    density_distribution: DistributionStats = field(default_factory=DistributionStats)
    low_quality_count: int = 0
    low_quality_rate: float = 0.0
    largest_clusters: list[dict[str, Any]] = field(default_factory=list)
    orphan_concepts: int = 0
    orphan_concept_rate: float = 0.0
    singleton_count: int = 0
    singleton_rate: float = 0.0
    cluster_type_distribution: dict[str, int] = field(default_factory=dict)
    health: StageHealth = field(default_factory=StageHealth)


@dataclass
class OpportunityEvaluation:
    total_opportunities: int = 0
    strong_pursue_count: int = 0
    worth_exploring_count: int = 0
    avg_opportunity_score: float = 0.0
    recommendation_distribution: dict[str, int] = field(default_factory=dict)
    business_model_distribution: dict[str, int] = field(default_factory=dict)
    has_opportunities: bool = False
    health: StageHealth = field(default_factory=StageHealth)


@dataclass
class TrendEvaluation:
    total_trends: int = 0
    growing_count: int = 0
    declining_count: int = 0
    emerging_count: int = 0
    anomaly_count: int = 0
    cross_platform_count: int = 0
    avg_trend_score: float = 0.0
    has_trends: bool = False
    health: StageHealth = field(default_factory=StageHealth)


@dataclass
class StageTiming:
    stage: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class GlobalEvaluation:
    documents: DocumentEvaluation = field(default_factory=DocumentEvaluation)
    observations: ObservationEvaluation = field(default_factory=ObservationEvaluation)
    evidence: EvidenceEvaluation = field(default_factory=EvidenceEvaluation)
    signals: SignalEvaluation = field(default_factory=SignalEvaluation)
    embeddings: EmbeddingEvaluation = field(default_factory=EmbeddingEvaluation)
    relationships: RelationshipEvaluation = field(default_factory=RelationshipEvaluation)
    clusters: ClusterEvaluation = field(default_factory=ClusterEvaluation)
    reasoning: ReasoningEvaluation = field(default_factory=ReasoningEvaluation)
    opportunities: OpportunityEvaluation = field(default_factory=OpportunityEvaluation)
    trends: TrendEvaluation = field(default_factory=TrendEvaluation)
    overall_health_score: float = 0.0
    worst_stage: str = ""
    all_warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    pipeline_timing: list[StageTiming] = field(default_factory=list)
    generated_at: str = ""
    evaluation_version: str = "1.0.0"
