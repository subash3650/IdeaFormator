"""Abstract base class for all data collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from pain_intelligence.ingestion.adapters.base import BaseAdapter
from pain_intelligence.ingestion.clients.base import HttpClient
from pain_intelligence.ingestion.config import CollectorConfig
from pain_intelligence.ingestion.models import SourceType, SyncState
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

COLLECTOR_VERSION = "1.0.0"


class BaseCollector(ABC):
    """Abstract base class for all data collectors.

    A collector is responsible ONLY for fetching raw data from a platform API.
    It does NOT normalize, validate, or persist. The Engine orchestrates the full
    pipeline: collect -> transform (adapter) -> normalize -> validate -> persist.

    Subclasses must implement:
    - authenticate(): Set up auth headers/tokens
    - health_check(): Verify API connectivity
    - fetch(state): Yield batches of raw API response dicts
    - adapter_class: The adapter class for this source
    """

    adapter_class: type[BaseAdapter]

    def __init__(self, config: CollectorConfig, client: HttpClient) -> None:
        self._config = config
        self._client = client
        self._api_calls: int = 0

    @property
    @abstractmethod
    def source(self) -> SourceType:
        """The platform this collector targets."""

    @abstractmethod
    def authenticate(self) -> None:
        """Set up authentication (headers, tokens) on the client."""

    @abstractmethod
    def health_check(self) -> bool:
        """Verify API connectivity. Returns True if healthy."""

    @abstractmethod
    def fetch(self, state: SyncState | None) -> Iterator[list[dict[str, Any]]]:
        """Fetch raw data from the API, yielding batches of raw response dicts.

        Must handle pagination internally. Each yield is one page/batch.
        Updates internal api_calls counter.
        """
