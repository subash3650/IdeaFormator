"""Persist pipeline stage — writes validated documents to JSONL and Parquet."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence.ingestion.models import RawDocument
from pain_intelligence.ingestion.storage.jsonl_store import JsonlStore
from pain_intelligence.ingestion.storage.parquet_store import ParquetStore
from pain_intelligence.ingestion.storage.manifest import ManifestBuilder
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


def _audit_field_types(documents: list[RawDocument]) -> dict[str, Counter]:
    """Audit all fields in RawDocuments and count types per field."""
    field_types: dict[str, Counter] = {}

    for doc in documents:
        flat = doc.to_flat_dict()
        for field_name, value in flat.items():
            if field_name not in field_types:
                field_types[field_name] = Counter()
            field_types[field_name][type(value).__name__] += 1

    return field_types


def _log_type_inconsistencies(field_types: dict[str, Counter], documents: list[RawDocument]) -> None:
    """Log fields with multiple concrete types and sample offending records."""
    inconsistent_fields = {name: counts for name, counts in field_types.items() if len(counts) > 1}

    if not inconsistent_fields:
        logger.info("Schema audit: all fields have consistent types")
        return

    logger.error("Schema audit: found {} fields with inconsistent types:", len(inconsistent_fields))

    for field_name, counts in inconsistent_fields.items():
        logger.error("Field: {}", field_name)
        logger.error("  Expected: one type, Found: {}", dict(counts))

        # Find first document with the minority type
        majority_type = counts.most_common(1)[0][0]
        for doc in documents:
            flat = doc.to_flat_dict()
            actual_type = type(flat[field_name]).__name__
            if actual_type != majority_type:
                logger.error("  Sample document_id: {}", doc.document_id)
                logger.error("  Sample source: {}", doc.source)
                logger.error("  Sample value type: {}", actual_type)
                logger.error("  Sample value: {}", repr(flat[field_name])[:100])
                break


class PersistStage:
    """Persists validated RawDocuments to disk as JSONL and Parquet.

    Output locations:
    - {output_base}/raw/{source}/{source}_{date}.jsonl  (immutable raw archive)
    - {output_base}/normalized/{source}/{source}_{date}.parquet  (pipeline format)
    """

    def __init__(self, output_base: Path) -> None:
        self._output_base = output_base
        self._jsonl_store = JsonlStore(output_base)
        self._parquet_store = ParquetStore(output_base)
        self._manifest_builder = ManifestBuilder(output_base)

    def run(
        self,
        source_name: str,
        documents: list[RawDocument],
    ) -> dict[str, str]:
        """Persist documents and return file paths.

        Returns dict with keys: jsonl_path, parquet_path, manifest_path.
        """
        if not documents:
            logger.warning("Persist stage: no documents to persist for {}", source_name)
            return {}

        # Schema audit: detect type inconsistencies before persistence
        field_types = _audit_field_types(documents)
        _log_type_inconsistencies(field_types, documents)

        # Write JSONL (immutable raw archive)
        jsonl_path = self._jsonl_store.write(source_name, documents)
        logger.info("Persisted {} documents to JSONL: {}", len(documents), jsonl_path)

        # Write Parquet (derived format for pipeline consumption)
        parquet_path = self._parquet_store.write(source_name, documents)
        logger.info("Persisted {} documents to Parquet: {}", len(documents), parquet_path)

        # Write manifest
        manifest_path = self._manifest_builder.write(
            source_name=source_name,
            documents=documents,
            file_paths=[str(jsonl_path), str(parquet_path)],
        )

        return {
            "jsonl_path": str(jsonl_path),
            "parquet_path": str(parquet_path),
            "manifest_path": str(manifest_path),
        }
