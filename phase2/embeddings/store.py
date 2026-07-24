"""Parquet-based embedding store for phase 2 assets."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phase2.embeddings.schema import EmbeddingRecord


class EmbeddingStore:
    """Reads and writes embedding records as Parquet files."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

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
        """Write a batch of records, overwriting any existing file."""
        path = self._source_path(source_type)
        rows = [self._record_to_row(r) for r in records]
        df = pl.DataFrame(rows, schema=self._schema())
        df.write_parquet(str(path))
        return path

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
        df.write_parquet(str(path))
        return path

    def read(self, source_type: str) -> pl.DataFrame:
        """Read embeddings for a source type."""
        path = self._source_path(source_type)
        if not path.exists():
            return pl.DataFrame(schema=self._schema())
        return pl.read_parquet(str(path))

    def read_all(self) -> pl.DataFrame:
        """Concatenate all source types into a single DataFrame."""
        dfs = []
        for st in ("observation", "evidence", "problem_signal"):
            path = self._source_path(st)
            if path.exists():
                dfs.append(pl.read_parquet(str(path)))
        return pl.concat(dfs, how="vertical") if dfs else pl.DataFrame(schema=self._schema())

    def exists(self, source_type: str) -> bool:
        return self._source_path(source_type).exists()