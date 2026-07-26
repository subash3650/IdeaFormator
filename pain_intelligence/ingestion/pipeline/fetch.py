"""Fetch pipeline stage — drives the collector to produce raw response batches."""

from __future__ import annotations

from typing import Any, Iterator

from pain_intelligence.ingestion.collectors.base import BaseCollector
from pain_intelligence.ingestion.models import SyncState
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class FetchStage:
    """Orchestrates a collector's fetch() method and yields raw batches.

    This stage handles:
    - Calling collector.authenticate() and health_check()
    - Iterating collector.fetch(state)
    - Counting API calls and pages
    """

    def run(
        self,
        collector: BaseCollector,
        state: SyncState | None,
    ) -> Iterator[tuple[list[dict[str, Any]], int]]:
        """Yield (batch, page_number) tuples from the collector.

        Handles authentication and health check before fetching.
        """
        try:
            collector.authenticate()
        except Exception as e:
            logger.error("Authentication failed: {}", e)
            return

        try:
            if not collector.health_check():
                logger.error("Health check failed for {}", collector.source.value)
                return
        except Exception as e:
            logger.error("Health check error: {}", e)
            return

        page = 0
        for batch in collector.fetch(state):
            page += 1
            logger.info(
                "[{}] Fetch stage: batch {} with {} items",
                collector.source.value,
                page,
                len(batch),
            )
            yield batch, page
