"""Data models for the embedding infrastructure."""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    observation = "observation"
    evidence = "evidence"
    problem_signal = "problem_signal"
    custom = "custom"


class EmbeddingProviderType(str, Enum):
    sentence_transformers = "sentence_transformers"
    ollama = "ollama"
    openai = "openai"


class EmbeddingRecord(BaseModel):
    embedding_id: str = Field(description="SHA-256 hex digest of source_id + provider + model + version")
    source_id: str = Field(description="Original row ID from source parquet")
    source_type: SourceType = Field(description="Which source asset this came from")
    provider: str = Field(description="Provider name (sentence_transformers, ollama, openai)")
    model: str = Field(description="Model name (e.g. all-MiniLM-L6-v2)")
    model_version: str | None = Field(None, description="Fingerprinted model version hash")
    dimension: int = Field(description="Embedding vector dimension")
    embedding: list[float] = Field(description="L2-normalized embedding vector")
    text_snippet: str | None = Field(None, description="Original text snippet, omitted if store_text=False")
    created_at: str = Field(description="ISO 8601 timestamp of generation")

    model_config = {"frozen": True, "extra": "forbid"}

    def to_vector(self) -> np.ndarray:
        return np.array(self.embedding, dtype=np.float32)


class SearchResult(BaseModel):
    embedding_id: str
    source_id: str
    source_type: SourceType
    provider: str
    model: str
    similarity: float
    text_snippet: str | None = None

    model_config = {"frozen": True, "extra": "forbid"}


class EmbeddingJob(BaseModel):
    source_type: SourceType
    source_path: Path
    output_path: Path
    batch_size: int = 64
    force: bool = False

    model_config = {"frozen": True, "extra": "forbid"}


class EmbeddingManifest(BaseModel):
    project: str = "pain-intelligence-engine"
    phase: str = "2"
    module: str = "embeddings"
    provider: str
    model: str
    model_version: str | None
    dimension: int
    normalize: bool
    num_vectors: int
    sources: dict[str, Any]
    created_at: str
    checksums: dict[str, str] | None = None

    model_config = {"extra": "forbid"}