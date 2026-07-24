"""Configuration models for the embedding engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from phase2.embeddings.schema import SourceType


class EmbeddingEngineConfig(BaseModel):
    provider: str = Field(default="sentence_transformers", description="Provider name")
    model: str = Field(default="all-MiniLM-L6-v2", description="Model name or path")
    model_version: str | None = Field(None, description="Explicit model version; auto-fingerprinted if None")
    dimension: int = Field(default=384, description="Expected embedding dimension")
    batch_size: int = Field(default=64, description="Texts per batch")
    device: str = Field(default="cpu", description="Torch device (cpu, cuda, mps)")
    normalize: bool = Field(default=True, description="L2-normalize output vectors")
    store_text: bool = Field(default=False, description="Include text snippet in EmbeddingRecord")
    cache_size: int = Field(default=0, description="Max cache entries (0 = unlimited)")
    output_dir: Path = Field(
        default=Path("pain_intelligence/knowledge/assets/phase2"),
        description="Root output directory for phase 2 assets",
    )
    source_paths: dict[SourceType, Path] = Field(
        default_factory=lambda: {
            SourceType.observation: Path("pain_intelligence/knowledge/assets/observations.parquet"),
            SourceType.evidence: Path("pain_intelligence/knowledge/assets/evidence.parquet"),
            SourceType.problem_signal: Path("pain_intelligence/knowledge/assets/problem_signals.parquet"),
        },
        description="Mapping of SourceType to input parquet paths",
    )
    concurrency: int = Field(default=1, description="Number of parallel workers (1 = sequential)")

    model_config = {"extra": "forbid", "frozen": True}


def load_embedding_config(path: str | Path) -> EmbeddingEngineConfig:
    """Load embedding config from a YAML file, falling back to defaults."""
    path = Path(path)
    if not path.exists():
        return EmbeddingEngineConfig()

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    embedding_cfg = raw.get("embedding", {})
    source_paths_raw = embedding_cfg.pop("source_paths", {})
    if source_paths_raw:
        parsed = {}
        for key, val in source_paths_raw.items():
            try:
                st = SourceType(key)
            except ValueError:
                st = SourceType.custom
            parsed[st] = Path(val)
        embedding_cfg["source_paths"] = parsed

    output_dir = embedding_cfg.get("output_dir")
    if output_dir is not None:
        embedding_cfg["output_dir"] = Path(output_dir)

    return EmbeddingEngineConfig(**embedding_cfg)