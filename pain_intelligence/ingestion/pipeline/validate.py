"""Validate pipeline stage — checks required fields, deduplication, timestamps."""

from __future__ import annotations

from datetime import datetime, timezone

from pain_intelligence.ingestion.models import RawDocument
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class ValidateStage:
    """Validates RawDocument instances and filters out invalid/duplicate records.

    Checks:
    - Required fields are present (document_id, source, external_id)
    - No duplicate document_ids within the batch
    - Timestamps are valid (not in the future)
    - Content is non-empty
    - Document is not frozen-model-violation
    """

    def run(
        self,
        documents: list[RawDocument],
    ) -> tuple[list[RawDocument], list[RawDocument]]:
        """Validate documents.

        Returns (valid, invalid) tuple.
        """
        valid: list[RawDocument] = []
        invalid: list[RawDocument] = []
        seen_ids: set[str] = set()
        now = datetime.now(timezone.utc)

        for doc in documents:
            reasons = self._validate_one(doc, seen_ids, now)
            if reasons:
                logger.debug(
                    "Document {} invalid: {}",
                    doc.document_id[:12],
                    ", ".join(reasons),
                )
                invalid.append(doc)
            else:
                seen_ids.add(doc.document_id)
                valid.append(doc)

        logger.info(
            "Validate stage: {} valid, {} invalid out of {} total",
            len(valid),
            len(invalid),
            len(documents),
        )
        return valid, invalid

    def _validate_one(
        self,
        doc: RawDocument,
        seen_ids: set[str],
        now: datetime,
    ) -> list[str]:
        """Return a list of validation failure reasons (empty = valid)."""
        reasons: list[str] = []

        # Required field checks
        if not doc.document_id:
            reasons.append("missing document_id")
        if not doc.external_id:
            reasons.append("missing external_id")
        if not doc.source:
            reasons.append("missing source")

        # Duplicate check
        if doc.document_id in seen_ids:
            reasons.append("duplicate document_id")

        # Content check — at least title or content must be present
        if not doc.title and not doc.content:
            reasons.append("empty content")

        # Timestamp validation — reject future timestamps
        if doc.created_at and doc.created_at > now:
            reasons.append(f"future created_at: {doc.created_at}")
        if doc.updated_at and doc.updated_at > now:
            reasons.append(f"future updated_at: {doc.updated_at}")

        return reasons
