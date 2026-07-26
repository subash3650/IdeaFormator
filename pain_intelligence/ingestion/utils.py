"""Utility functions for the ingestion framework."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any


def compute_checksum(content: str) -> str:
    """Compute the SHA-256 checksum of string content."""
    if not content:
        return ""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_document_id(source: str, external_id: str) -> str:
    """Compute a deterministic document ID from source and external ID."""
    raw = f"{source}:{external_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_timestamp(value: Any) -> datetime | None:
    """Parse a timestamp into a UTC datetime object."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)

    # Clean string format
    s = str(value).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            # Handle standard ISO Z suffix which datetime.strptime does not always support on older python
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except ValueError:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None
