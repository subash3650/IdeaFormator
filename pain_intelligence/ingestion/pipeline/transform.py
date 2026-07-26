"""Transform pipeline stage — dispatches raw batches through the adapter."""

from __future__ import annotations

from typing import Any

from pain_intelligence.ingestion.adapters.base import BaseAdapter
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class TransformStage:
    """Transforms raw API response batches into normalized intermediate dicts.

    Uses the source-specific adapter to handle API format differences.
    """

    def run(
        self,
        adapter: BaseAdapter,
        raw_batch: list[dict[str, Any]],
        page: int = 0,
    ) -> list[dict[str, Any]]:
        """Transform a raw batch through the adapter.

        Returns a list of normalized intermediate dicts.
        """
        transformed = adapter.transform_batch(raw_batch)
        logger.info(
            "[{}] Transform stage: {} -> {} items (page {})",
            adapter.source.value,
            len(raw_batch),
            len(transformed),
            page,
        )
        return transformed
