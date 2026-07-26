"""Configuration models for the Semantic Clustering Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class QualityWeights(BaseModel):
    """Configurable weights for quality score computation."""

    cohesion: float = Field(default=0.35, ge=0.0, le=1.0, description="Internal cohesion weight")
    density: float = Field(default=0.25, ge=0.0, le=1.0, description="Density weight")
    separation: float = Field(default=0.20, ge=0.0, le=1.0, description="External separation weight")
    connectivity: float = Field(default=0.20, ge=0.0, le=1.0, description="Connectivity weight")

    model_config = {"extra": "forbid", "frozen": True}

    def normalize(self) -> QualityWeights:
        """Return normalized weights summing to 1.0."""
        total = self.cohesion + self.density + self.separation + self.connectivity
        if total == 0:
            return QualityWeights()
        return QualityWeights(
            cohesion=self.cohesion / total,
            density=self.density / total,
            separation=self.separation / total,
            connectivity=self.connectivity / total,
        )


class ClusteringConfig(BaseModel):
    """Configuration for the clustering engine."""

    provider: str = Field(default="connected_components", description="Clustering provider name")
    minimum_cluster_size: int = Field(default=3, ge=1, description="Minimum cluster size")
    maximum_cluster_size: int = Field(default=500, ge=1, description="Maximum cluster size")
    relationship_threshold: float = Field(default=0.82, ge=0.0, le=1.0, description="Minimum similarity for edge inclusion")
    quality_threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="Quality threshold for LOW_QUALITY classification")
    remove_singletons: bool = Field(default=True, description="Remove singleton clusters")
    merge_small_clusters: bool = Field(default=True, description="Attempt to merge clusters below minimum size")
    deterministic: bool = Field(default=True, description="Enforce deterministic outputs")
    output_directory: Path = Field(
        default=Path("pain_intelligence/knowledge/assets/phase2"),
        description="Output directory for cluster assets",
    )
    quality_weights: QualityWeights = Field(default_factory=QualityWeights, description="Quality score component weights")
    version: str = Field(default="1.0", description="Engine version")

    model_config = {"extra": "forbid", "frozen": True}

    @property
    def config_hash(self) -> str:
        """Deterministic hash of clustering configuration."""
        import hashlib
        import json
        config_dict = self.model_dump(mode="json")
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def load_clustering_config(path: str | Path) -> ClusteringConfig:
    """Load clustering config from YAML file, falling back to defaults."""
    path = Path(path)
    if not path.exists():
        return ClusteringConfig()

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    clustering_cfg = raw.get("clustering", {})
    output_dir = clustering_cfg.get("output_directory")
    if output_dir is not None:
        clustering_cfg["output_directory"] = Path(output_dir)

    quality_weights_raw = clustering_cfg.pop("quality_weights", {})
    if quality_weights_raw:
        clustering_cfg["quality_weights"] = QualityWeights(**quality_weights_raw)

    return ClusteringConfig(**clustering_cfg)