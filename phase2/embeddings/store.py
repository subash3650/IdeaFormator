"""Parquet-based embedding store for phase 2 assets."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pain_intelligence import PIPELINE_VERSION, SCHEMA_VERSION
from pain_intelligence.knowledge.metadata import (
    make_asset_metadata,
    read_parquet_metadata,
    write_parquet_with_metadata,
)
from phase2.embeddings.schema import EmbeddingRecord


class EmbeddingStore:
    """Reads and writes embedding records as Parquet files.

    ALWAYS overwrites existing files to prevent stale data.
    Supports embedded metadata (run_id, checksums).
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._run_id: str = ""
        self._input_checksum: str = ""
        self._input_document_count: int = 0

    def set_run_metadata(
        self,
        run_id: str,
        input_checksum: str = "",
        input_document_count: int = 0,
    ) -> None:
        self._run_id = run_id
        self._input_checksum = input_checksum
        self._input_document_count = input_document_count

    def _source_path(self, source_type: str) -> Path:
        return self._base_path / f"embeddings_{source_type}.parquet"

    @staticmethod
    def _record_to_row(rec: EmbeddingRecord) -> dict:
        return {
            "embedding_id": rec.embedding_id,
            "source_id": rec.source_id,
            "source_type": rec.source_type.value,
            "provider": rec.provider,
            "model": rec.model,
            "model_version": rec.model_version,
            "dimension": rec.dimension,
            "embedding": rec.embedding,
            "text_snippet": rec.text_snippet,
            "created_at": rec.created_at,
        }

    @staticmethod
    def _schema() -> dict[str, type]:
        return {
            "embedding_id": str,
            "source_id": str,
            "source_type": str,
            "provider": str,
            "model": str,
            "model_version": str,
            "dimension": int,
            "embedding": list[float],
            "text_snippet": str,
            "created_at": str,
        }

    def write(self, records: list[EmbeddingRecord], source_type: str) -> Path:
        """Write a batch of records, ALWAYS overwriting."""
        path = self._source_path(source_type)
        metadata = make_asset_metadata(
            run_id=self._run_id,
            input_checksum=self._input_checksum,
            input_document_count=self._input_document_count,
            record_count=len(records),
            source_type=source_type,
        )

        if not records:
            df = pl.DataFrame(schema=self._schema())
        else:
            rows = [self._record_to_row(r) for r in records]
            df = pl.DataFrame(rows, schema=self._schema())

        return write_parquet_with_metadata(df, path, metadata=metadata)

    def append(self, records: list[EmbeddingRecord], source_type: str) -> Path:
        """Append records to an existing parquet file or create a new one."""
        path = self._source_path(source_type)
        rows = [self._record_to_row(r) for r in records]
        new_df = pl.DataFrame(rows, schema=self._schema())
        if path.exists():
            existing = pl.read_parquet(str(path))
            df = pl.concat([existing, new_df], how="vertical")
        else:
            df = new_df
        metadata = make_asset_metadata(
            run_id=self._run_id,
            input_checksum=self._input_checksum,
            input_document_count=self._input_document_count,
            record_count=df.height,
            source_type=source_type,
        )
        return write_parquet_with_metadata(df, path, metadata=metadata)

    def read(self, source_type: str) -> pl.DataFrame:
        path = self._source_path(source_type)
        if not path.exists():
            return pl.DataFrame(schema=self._schema())
        return pl.read_parquet(str(path))

    def read_all(self) -> pl.DataFrame:
        dfs = []
        for st in ("observation", "evidence", "problem_signal"):
            path = self._source_path(st)
            if path.exists():
                dfs.append(pl.read_parquet(str(path)))
        return pl.concat(dfs, how="vertical") if dfs else pl.DataFrame(schema=self._schema())

    def exists(self, source_type: str) -> bool:
        return self._source_path(source_type).exists()

    def get_asset_metadata(self, source_type: str) -> dict[str, str]:
        path = self._source_path(source_type)
        return read_parquet_metadata(path)
