"""Unified document schema for the Pain Intelligence Engine.

Every dataset, regardless of source platform, is transformed into this schema.
This is the single source of truth for all downstream processing.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class Platform(str, Enum):
    """Supported data source platforms."""

    AMAZON = "amazon"
    YELP = "yelp"
    TWITTER = "twitter"
    REDDIT = "reddit"
    UNKNOWN = "unknown"


class RemovalReason(str, Enum):
    """Reasons a document was removed during preprocessing."""

    EMPTY_TEXT = "empty_text"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    DUPLICATE = "duplicate"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    ENCODING_ERROR = "encoding_error"
    MISSING_TEXT = "missing_text"


class RemovedDocument(BaseModel):
    """A document removed during preprocessing, preserved for auditability."""

    document_id: str
    platform: Platform
    source_dataset: str
    text_preview: str = ""
    reason: RemovalReason
    original_length: int = 0


class Document(BaseModel):
    """Unified document representation across all platforms.

    Every dataset loader normalizes its records into this schema.
    Downstream consumers (embeddings, clustering, emotion detection)
    operate exclusively on this model.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    platform: Platform
    source_dataset: str
    title: str | None = None
    text: str
    rating: float | None = None
    author: str | None = None
    country: str | None = None
    location: str | None = None
    language: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_record: dict[str, Any] = Field(default_factory=dict)
    clean_text: str | None = None
    document_length: int = 0

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: float | None) -> float | None:
        """Normalize rating to 0.0-5.0 range."""
        if v is None:
            return None
        if v < 0.0:
            return 0.0
        if v > 5.0:
            return 5.0
        return round(v, 1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is non-empty after stripping."""
        if not v or not v.strip():
            raise ValueError("Document text cannot be empty")
        return v

    def to_flat_dict(self) -> dict[str, Any]:
        """Serialize to a flat dictionary suitable for Parquet/CSV output."""
        d = self.model_dump()
        d["metadata"] = str(d["metadata"])
        d["raw_record"] = str(d["raw_record"])
        d["created_at"] = d["created_at"].isoformat() if d["created_at"] else None
        return d

    @property
    def effective_text(self) -> str:
        """Return clean_text if available, otherwise raw text."""
        return self.clean_text if self.clean_text else self.text
