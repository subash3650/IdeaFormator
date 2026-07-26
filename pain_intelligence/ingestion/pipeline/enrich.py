"""Enrich pipeline stage — adds computed metadata like language detection."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pain_intelligence.ingestion.models import RawDocument
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

# Lightweight language detection without heavy dependencies
try:
    from langdetect import detect as _detect_lang

    def _detect_language(text: str) -> str | None:
        try:
            return _detect_lang(text)
        except Exception:
            return None

except ImportError:
    # Fallback: no language detection available
    def _detect_language(text: str) -> str | None:  # type: ignore[misc]
        return None


_INSPECTED_FIELDS = [
    "source",
    "source_type",
    "external_id",
    "title",
    "content",
    "author",
    "created_at",
    "updated_at",
    "language",
    "url",
    "tags",
    "categories",
    "metadata",
    "raw_json",
    "checksum",
]


class SchemaValidationError(Exception):
    """Raised when RawDocument fields have non-deterministic types."""


class EnrichStage:
    """Enriches RawDocument instances with computed metadata.

    Currently enriches:
    - Language detection (if langdetect is available and language is not set)
    - Content length metadata
    - Schema validation for deterministic field types
    """

    def run(self, documents: list[RawDocument]) -> list[RawDocument]:
        """Enrich documents in-place is not possible (frozen), so we rebuild."""
        enriched: list[RawDocument] = []

        for doc in documents:
            new_metadata = dict(doc.metadata)

            # Language detection if not already set
            language = doc.language
            if not language and doc.content:
                text_sample = (doc.title or "") + " " + (doc.content or "")
                language = _detect_language(text_sample[:500])
                if language:
                    new_metadata["detected_language"] = language

            # Content length metadata
            if doc.content:
                new_metadata["content_length"] = len(doc.content)
            if doc.title:
                new_metadata["title_length"] = len(doc.title)

            # Rebuild the document with enriched metadata
            enriched_doc = doc.model_copy(update={
                "language": language or doc.language,
                "metadata": new_metadata,
            })
            enriched.append(enriched_doc)

        # Schema validation: detect type inconsistencies
        self._validate_schema(enriched)

        logger.info("Enrich stage: {} documents enriched", len(enriched))
        return enriched

    def _validate_schema(self, documents: list[RawDocument]) -> None:
        """Validate that all RawDocuments have deterministic field types.

        Checks each field for type consistency across all documents.
        Logs warnings for fields with multiple concrete types.
        Raises SchemaValidationError if inconsistencies are detected.
        """
        field_types: dict[str, Counter] = {}
        for doc in documents:
            raw = doc.model_dump()
            for field_name in _INSPECTED_FIELDS:
                if field_name not in field_types:
                    field_types[field_name] = Counter()
                value = raw.get(field_name)
                field_types[field_name][type(value).__name__] += 1

        for field_name, counts in field_types.items():
            non_null_types = {t: c for t, c in counts.items() if t != "NoneType"}
            if len(non_null_types) > 1:
                logger.warning(
                    "Schema: field '{}' has non-deterministic types: {}",
                    field_name, dict(non_null_types),
                )
                # Find first offending document
                types_list = list(non_null_types.keys())
                for doc in documents:
                    raw = doc.model_dump()
                    actual = type(raw.get(field_name)).__name__
                    if actual in types_list:
                        logger.warning(
                            "  Offending document_id={}, source={}, field={}, type={}",
                            doc.document_id, doc.source, field_name, actual,
                        )
                        break
