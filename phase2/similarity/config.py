"""Configuration models for the Semantic Relationship Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from phase2.embeddings.schema import SourceType


class SimilarityEngineConfig(BaseModel):
    """Configuration for the similarity engine."""

    metric: str = Field(default="cosine", description="cosine, dot_product, euclidean")
    similarity_threshold: float = Field(default=0.82, ge=0.0, le=1.0, description="Minimum similarity score")
    top_k: int = Field(default=20, ge=1, description="Max neighbors per source item")
    batch_size: int = Field(default=1024, ge=1, description="Batch size for vector operations")
    normalize_scores: bool = Field(default=True, description="Normalize scores to [0, 1]")
    minimum_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold (downstream use; engine does not filter on confidence)")
    store_bidirectional: bool = Field(default=False, description="Store both directions of each relationship")
    output_directory: Path = Field(
        default=Path("pain_intelligence/knowledge/assets/phase2"),
        description="Output directory for relationship assets",
    )
    version: str = Field(default="1.0", description="Schema/engine version")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    embedding_provider: str = Field(default="sentence_transformers", description="Embedding provider name")
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")
    allowed_relationships: dict[SourceType, list[SourceType]] = Field(
        default_factory=lambda: {
            SourceType.observation: [SourceType.observation, SourceType.evidence, SourceType.problem_signal],
            SourceType.evidence: [SourceType.evidence, SourceType.problem_signal],
            SourceType.problem_signal: [SourceType.problem_signal],
        },
        description="Allowed cross-type relationships per source",
    )
    source_paths: dict[SourceType, Path] = Field(
        default_factory=lambda: {
            SourceType.observation: Path("pain_intelligence/knowledge/assets/phase2/embeddings_observation.parquet"),
            SourceType.evidence: Path("pain_intelligence/knowledge/assets/phase2/embeddings_evidence.parquet"),
            SourceType.problem_signal: Path("pain_intelligence/knowledge/assets/phase2/embeddings_problem_signal.parquet"),
        },
        description="Mapping of SourceType to embedding parquet paths",
    )
    concurrency: int = Field(default=1, description="Number of parallel workers")

    model_config = {"extra": "forbid", "frozen": True}

    @property
    def model_fingerprint(self) -> str:
        """Deterministic model fingerprint: provider/model@dimension."""
        return f"{self.embedding_provider}/{self.embedding_model}@{self.embedding_dimension}d"


def load_similarity_config(path: str | Path) -> SimilarityEngineConfig:
    """Load similarity config from a YAML file, falling back to defaults."""
    path = Path(path)
    if not path.exists():
        return SimilarityEngineConfig()

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    similarity_cfg = raw.get("similarity", {})
    source_paths_raw = similarity_cfg.pop("source_paths", {})
    if source_paths_raw:
        parsed = {}
        for key, val in source_paths_raw.items():
            try:
                st = SourceType(key)
            except ValueError:
                continue
            parsed[st] = Path(val)
        similarity_cfg["source_paths"] = parsed

    allowed_raw = similarity_cfg.pop("allowed_relationships", {})
    if allowed_raw:
        parsed = {}
        for key, val in allowed_raw.items():
            try:
                st = SourceType(key)
            except ValueError:
                continue
            parsed[st] = [SourceType(v) for v in val if v in {e.value for e in SourceType}]
        similarity_cfg["allowed_relationships"] = parsed

    output_dir = similarity_cfg.get("output_directory")
    if output_dir is not None:
        similarity_cfg["output_directory"] = Path(output_dir)

    return SimilarityEngineConfig(**similarity_cfg)
