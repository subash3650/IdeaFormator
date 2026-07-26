"""Parquet storage for pipeline-consumable normalized documents."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from pain_intelligence.ingestion.models import RawDocument
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

# Define Parquet schema for consistent column types
PARQUET_SCHEMA = {
    "document_id": pl.Utf8,
    "source": pl.Utf8,
    "source_type": pl.Utf8,
    "external_id": pl.Utf8,
    "title": pl.Utf8,
    "content": pl.Utf8,
    "author": pl.Utf8,
    "created_at": pl.Utf8,
    "updated_at": pl.Utf8,
    "language": pl.Utf8,
    "url": pl.Utf8,
    "tags": pl.Utf8,
    "categories": pl.Utf8,
    "metadata": pl.Utf8,
    "raw_json": pl.Utf8,
    "checksum": pl.Utf8,
    "schema_version": pl.Utf8,
    "pipeline_version": pl.Utf8,
    "collector_version": pl.Utf8,
    "ingested_at": pl.Utf8,
}


class ParquetStore:
    """Writes RawDocuments as Parquet files for pipeline consumption.

    Output path: {output_base}/normalized/{source}/{source}_{YYYYMMDD}.parquet
    """

    def __init__(self, output_base: Path) -> None:
        self._output_base = output_base

    def write(self, source_name: str, documents: list[RawDocument]) -> Path:
        """Write documents to a Parquet file (overwrite)."""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        source_dir = self._output_base / "normalized" / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        path = source_dir / f"{source_name}_{date_str}.parquet"

        # Convert to flat dicts
        rows = [doc.to_flat_dict() for doc in documents]

        # Build DataFrame with explicit schema
        df = pl.DataFrame(rows)

        # Ensure all expected columns exist (fill missing with null)
        for col, dtype in PARQUET_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

        # Select columns in canonical order
        ordered_cols = [col for col in PARQUET_SCHEMA if col in df.columns]
        df = df.select(ordered_cols)

        df.write_parquet(path)
        logger.debug("Wrote {} documents to {}", len(documents), path)
        return path

    def read(self, source_name: str, date_str: str | None = None) -> pl.DataFrame:
        """Read documents from a Parquet file."""
        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

        path = self._output_base / "normalized" / source_name / f"{source_name}_{date_str}.parquet"
        if not path.exists():
            return pl.DataFrame()

        return pl.read_parquet(path)
