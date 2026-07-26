"""Data models for the ingestion framework."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ConfigurationError(Exception):
    """Raised when a collector lacks required credentials or configuration."""
    pass


class SourceType(str, Enum):
    GITHUB = "github"
    HACKERNEWS = "hackernews"
    PRODUCTHUNT = "producthunt"
    YOUTUBE = "youtube"
    PLAYSTORE = "playstore"


class RawDocument(BaseModel):
    """Normalized document representation across all ingestion collectors."""

    model_config = {"frozen": True, "extra": "forbid"}

    schema_version: str = "1.0.0"
    document_id: str  # SHA-256 of (source + external_id)
    source: SourceType
    source_type: str  # e.g., "issue", "comment", "story", "review"
    external_id: str  # Platform-specific unique ID
    title: str | None = None
    content: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    language: str | None = None
    url: str | None = None
    tags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_json: dict[str, Any] = Field(default_factory=dict)
    checksum: str = ""  # SHA-256 hash of content/title
    pipeline_version: str = "0.1.0"
    collector_version: str = "1.0.0"
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for tabular outputs (Parquet).

        Normalizes all nullable fields to deterministic types so that
        Polars does not infer Null columns when the first N rows are all-None.
        """
        d = self.model_dump()
        d["metadata"] = str(d["metadata"])
        d["raw_json"] = str(d["raw_json"])
        d["tags"] = ",".join(d["tags"])
        d["categories"] = ",".join(d["categories"])
        d["created_at"] = d["created_at"].isoformat() if d["created_at"] else ""
        d["updated_at"] = d["updated_at"].isoformat() if d["updated_at"] else ""
        d["ingested_at"] = d["ingested_at"].isoformat() if d["ingested_at"] else ""

        # Normalize nullable string fields to ensure Polars infers String type
        # even when the first N rows are all-None.
        d["title"] = d["title"] or ""
        d["content"] = d["content"] or ""
        d["author"] = d["author"] or ""
        d["language"] = d["language"] or ""
        d["url"] = d["url"] or ""

        d["source"] = d["source"].value
        return d


class CollectionResult(BaseModel):
    """Stats and results of an ingestion run per collector."""

    model_config = {"frozen": True}

    source: SourceType
    documents_collected: int = 0
    documents_valid: int = 0
    documents_duplicate: int = 0
    documents_invalid: int = 0
    documents_failed: int = 0
    pages_fetched: int = 0
    api_calls: int = 0
    errors: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class SyncState(BaseModel):
    """Maintains progress cursor/etag/next_page for incremental syncs."""

    model_config = {"frozen": True}

    source: SourceType
    last_sync: datetime | None = None
    cursor: str | None = None
    etag: str | None = None
    next_page_token: str | None = None
    failure_count: int = 0
    last_error: str | None = None
    total_collected: int = 0
    rate_limit_remaining: int | None = None
    rate_limit_reset: datetime | None = None


class IngestionManifest(BaseModel):
    """Manifest describing generated batch/run results."""

    model_config = {"frozen": True}

    schema_version: str = "1.0.0"
    pipeline_version: str = "0.1.0"
    collector_version: str = "1.0.0"
    api_version: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: SourceType
    document_count: int = 0
    checksum: str = ""
    file_paths: list[str] = Field(default_factory=list)
