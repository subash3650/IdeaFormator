"""Normalize pipeline stage — converts intermediate dicts to RawDocument models."""

from __future__ import annotations

from typing import Any

from pain_intelligence.ingestion.models import RawDocument
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class NormalizeStage:
    """Converts normalized intermediate dicts (from adapters) into RawDocument instances.

    This is a strict schema enforcement step: every field is validated
    by the Pydantic model.
    """

    def run(self, records: list[dict[str, Any]]) -> list[RawDocument]:
        """Normalize a list of intermediate dicts into RawDocument instances.

        Returns only successfully created documents.
        """
        documents: list[RawDocument] = []
        errors = 0

        for record in records:
            try:
                doc = RawDocument(**record)
                documents.append(doc)
            except Exception as e:
                errors += 1
                logger.debug("Normalization failed for record {}: {}", record.get("document_id", "?"), e)

        if errors:
            logger.warning("Normalize stage: {} documents failed out of {}", errors, len(records))
        else:
            logger.info("Normalize stage: {} documents created", len(documents))

        return documents
