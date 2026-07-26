"""Configuration for the Opportunity Discovery Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from phase3.opportunity.schema import MarketSize, RankingStrategy, ScoreWeights


class OpportunityConfig(BaseModel):
    """Configuration for the Opportunity Discovery Engine."""

    model_config = {"frozen": True, "extra": "forbid"}

    output_dir: Path
    knowledge_dir: Path | None = Field(default=None)

    # Scoring
    scoring_provider: str = Field(default="weighted")
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)
    enabled_scoring_providers: list[str] = Field(
        default_factory=lambda: ["weighted", "market", "trend", "competition"],
    )

    # Business model
    enabled_business_model_providers: list[str] = Field(
        default_factory=lambda: [
            "saas",
            "ai_agent",
            "marketplace",
            "chrome_extension",
            "api",
            "mobile_app",
            "b2b_platform",
            "developer_tool",
            "consumer_product",
        ],
    )

    # Ranking
    ranking_strategy: RankingStrategy = Field(default=RankingStrategy.COMPOSITE)
    top_k: int = Field(default=20, ge=1, le=500)
    dedup_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # Extraction
    min_evidence_for_opportunity: int = Field(default=2, ge=1)
    min_confidence_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    min_cluster_size_for_opportunity: int = Field(default=2, ge=1)

    # Recommendation
    auto_recommend: bool = Field(default=True)
    strong_pursue_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    worth_exploring_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    niche_potential_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    monitor_threshold: float = Field(default=0.10, ge=0.0, le=1.0)

    # Market estimation
    market_size_method: str = Field(default="evidence_based")
    cross_platform_bonus: float = Field(default=0.10, ge=0.0, le=1.0)

    # Cache
    cache_enabled: bool = Field(default=True)

    # Confidence
    confidence_method: str = Field(default="weighted_average")
    reasoning_confidence_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    evidence_confidence_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    graph_confidence_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    market_confidence_weight: float = Field(default=0.15, ge=0.0, le=1.0)

    # Versioning
    version: str = Field(default="1.0")
    opportunity_version: str = Field(default="1.0")

    @property
    def opportunity_dir(self) -> Path:
        base = self.knowledge_dir or self.output_dir
        return base / "opportunity"


def load_opportunity_config(path: str | Path) -> OpportunityConfig:
    """Load opportunity config from YAML file, falling back to defaults."""
    path = Path(path)
    default_output = Path("pain_intelligence/knowledge/assets/phase3")
    if not path.exists():
        return OpportunityConfig(output_dir=default_output)

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    opp_cfg = raw.get("opportunity", {})

    output_dir = opp_cfg.get("output_dir")
    if output_dir is not None:
        opp_cfg["output_dir"] = Path(output_dir)
    else:
        output_dir_raw = raw.get("output_directory")
        if output_dir_raw is not None:
            opp_cfg["output_dir"] = Path(output_dir_raw)
        else:
            opp_cfg["output_dir"] = default_output

    knowledge_dir = opp_cfg.get("knowledge_dir")
    if knowledge_dir is not None:
        opp_cfg["knowledge_dir"] = Path(knowledge_dir)

    sw_raw = opp_cfg.pop("score_weights", None)
    if sw_raw:
        opp_cfg["score_weights"] = ScoreWeights(**sw_raw)

    for enum_field, enum_cls in [
        ("ranking_strategy", RankingStrategy),
        ("market_size_method", MarketSize),
    ]:
        val = opp_cfg.pop(enum_field, None)
        if val is not None:
            opp_cfg[enum_field] = enum_cls(val)

    return OpportunityConfig(**opp_cfg)
