"""Configuration for the Trend Intelligence Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from phase3.trend.schema import TrendScoreWeights


class TrendConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    output_dir: Path = Field(description="Base output directory for trend assets")
    knowledge_dir: Path | None = Field(default=None, description="Knowledge base directory (overrides output_dir for snapshots)")

    # Detection
    min_growth_pct: float = Field(default=5.0, ge=0.0, description="Minimum growth % to qualify as a trend")
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence for trend detection")
    min_snapshots: int = Field(default=2, ge=1, description="Minimum snapshots required for analysis")
    window_size: int = Field(default=7, ge=1, description="Default sliding window in days")
    seasonality_period: int = Field(default=7, ge=1, description="Default seasonality period in days")
    anomaly_threshold: float = Field(default=2.0, ge=0.0, description="Z-score threshold for anomaly detection")
    comparison_window: int = Field(default=1, ge=1, description="Number of snapshots back to compare (1 = immediate prior)")

    # Scoring
    score_weights: TrendScoreWeights = Field(default_factory=TrendScoreWeights)
    enabled_scoring_providers: list[str] | None = Field(default=None)

    # Cache
    cache_enabled: bool = Field(default=True)

    # Ranking
    top_k: int = Field(default=20, ge=1, le=500)

    # Version
    version: str = Field(default="1.0")

    @field_validator("output_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("knowledge_dir", mode="before")
    @classmethod
    def _coerce_knowledge_path(cls, v: Any) -> Path | None:
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @property
    def trend_dir(self) -> Path:
        base = self.knowledge_dir or self.output_dir
        return Path(base) / "trend"


def load_trend_config(path: str | Path | None = None) -> TrendConfig:
    if path is None:
        return TrendConfig(output_dir=Path("pain_intelligence/knowledge/assets/phase3"))

    path = Path(path)
    if not path.exists():
        return TrendConfig(output_dir=Path("pain_intelligence/knowledge/assets/phase3"))

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "trend" not in raw:
        return TrendConfig(output_dir=Path("pain_intelligence/knowledge/assets/phase3"))

    cfg = raw["trend"]
    if "output_dir" in cfg:
        cfg["output_dir"] = Path(str(cfg["output_dir"]))
    if "knowledge_dir" in cfg and cfg["knowledge_dir"]:
        cfg["knowledge_dir"] = Path(str(cfg["knowledge_dir"]))
    return TrendConfig(**cfg)
