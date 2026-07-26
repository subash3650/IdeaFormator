"""Base adapter for transforming platform-specific API responses into a common dict format."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pain_intelligence.ingestion.models import SourceType


class BaseAdapter(ABC):
    """Abstract base class for source-specific data adapters.

    An adapter transforms a raw API response dict (from the collector)
    into a normalized intermediate dict that can be passed to the pipeline
    for RawDocument creation.

    If the upstream API changes, only the adapter changes.
    """

    @abstractmethod
    def transform(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Transform a single raw API response into a normalized dict.

        Returns a dict with keys matching RawDocument fields.
        """

    @abstractmethod
    def transform_batch(self, raw_responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform a batch of raw API responses into normalized dicts."""

    @property
    @abstractmethod
    def source(self) -> SourceType:
        """The platform/source this adapter handles."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Adapter version identifier."""
