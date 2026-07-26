"""Utilities for embedding and reading Parquet file metadata.

Adds run_id, pipeline_version, schema_version, generated_at,
input_checksum, and input_document_count to every generated asset.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence import PIPELINE_VERSION, SCHEMA_VERSION

METADATA_KEY = b"pipeline_metadata"


def make_asset_metadata(
    run_id: str,
    input_checksum: str = "",
    input_document_count: int = 0,
    record_count: int = 0,
    **extra: str | int,
) -> dict[str, str]:
    """Build the standard metadata dict for a pipeline asset."""
    return {
        "run_id": run_id,
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_checksum": input_checksum,
        "input_document_count": str(input_document_count),
        "record_count": str(record_count),
        **{k: str(v) for k, v in extra.items()},
    }


def write_parquet_with_metadata(
    df: pl.DataFrame,
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a Polars DataFrame to Parquet with embedded key-value metadata.

    If the DataFrame is empty, writes an empty Parquet file with the
    correct schema and metadata.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to PyArrow table
    if df.height == 0 and df.width == 0:
        table = pa.Table.from_pylist([], schema=pa.schema([]))
    else:
        table = df.to_arrow()

    # Attach metadata
    if metadata:
        existing = table.schema.metadata or {}
        existing[METADATA_KEY] = json.dumps(metadata).encode("utf-8")
        table = table.replace_schema_metadata(existing)

    pq.write_table(table, str(path))
    return path


def read_parquet_metadata(path: str | Path) -> dict[str, str]:
    """Read embedded metadata from a Parquet file.

    Returns an empty dict if no metadata is found.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.exists():
        return {}

    try:
        schema = pq.read_schema(str(path))
        if schema.metadata and METADATA_KEY in schema.metadata:
            return json.loads(schema.metadata[METADATA_KEY].decode("utf-8"))
    except Exception:
        pass
    return {}


def get_run_id_from_asset(path: str | Path) -> str:
    """Extract the run_id from an asset's embedded metadata."""
    meta = read_parquet_metadata(path)
    return meta.get("run_id", "")
