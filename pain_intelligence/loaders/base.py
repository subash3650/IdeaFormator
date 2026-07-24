"""Abstract base class for all dataset loaders.

Every platform loader inherits from BaseLoader and implements
its own schema understanding. Column mapping is always by NAME,
never by index.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from pain_intelligence.schema.document import Document, Platform


class BaseLoader(ABC):
    """Abstract base for dataset loaders.

    Subclasses must implement:
    - platform: Class-level Platform enum value.
    - column_mapping: Dict mapping source column names to Document fields.
    - _detect(df_columns: list[str]) -> bool: Column-name-based detection.
    - _transform_row(row: dict) -> Document: Transform a single record.
    """

    platform: Platform
    source_name: str

    @abstractmethod
    def _detect(self, df_columns: list[str]) -> bool:
        """Detect if this loader handles the given column schema.

        Args:
            df_columns: List of column names from the dataset.

        Returns:
            True if this loader can handle the dataset.
        """
        ...

    @abstractmethod
    def _transform_row(self, row: dict[str, Any]) -> Document:
        """Transform a raw row dict into a Document.

        Args:
            row: Dictionary with column_name -> value.

        Returns:
            A Document instance.
        """
        ...

    def load(
        self,
        file_path: str | Path,
        chunk_size: int = 50_000,
    ) -> Iterator[pl.DataFrame]:
        """Load a dataset file in chunks.

        Uses Polars for memory-efficient reading. Supports CSV, Parquet,
        JSON, and JSONL formats.

        Args:
            file_path: Path to the dataset file.
            chunk_size: Number of rows per chunk.

        Yields:
            DataFrame chunks of at most chunk_size rows.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            yield from self._load_csv(path, chunk_size)
        elif suffix == ".parquet":
            yield from self._load_parquet(path, chunk_size)
        elif suffix == ".json":
            yield from self._load_json(path, chunk_size)
        elif suffix == ".jsonl":
            yield from self._load_jsonl(path, chunk_size)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_csv(
        self, path: Path, chunk_size: int
    ) -> Iterator[pl.DataFrame]:
        """Load CSV in chunks using Polars streaming."""
        try:
            df = pl.read_csv(
                path,
                infer_schema_length=10000,
                truncate_ragged_lines=True,
                encoding="utf8-lossy",
            )
        except Exception:
            df = pl.read_csv(
                path,
                infer_schema_length=1000,
                truncate_ragged_lines=True,
                has_header=True,
                separator=",",
            )

        total_rows = len(df)
        for start in range(0, total_rows, chunk_size):
            yield df.slice(start, min(chunk_size, total_rows - start))

    def _load_parquet(
        self, path: Path, chunk_size: int
    ) -> Iterator[pl.DataFrame]:
        """Load Parquet in chunks."""
        df = pl.read_parquet(path)
        total_rows = len(df)
        for start in range(0, total_rows, chunk_size):
            yield df.slice(start, min(chunk_size, total_rows - start))

    def _load_json(
        self, path: Path, chunk_size: int
    ) -> Iterator[pl.DataFrame]:
        """Load JSON in chunks."""
        df = pl.read_json(path)
        total_rows = len(df)
        for start in range(0, total_rows, chunk_size):
            yield df.slice(start, min(chunk_size, total_rows - start))

    def _load_jsonl(
        self, path: Path, chunk_size: int
    ) -> Iterator[pl.DataFrame]:
        """Load JSONL (newline-delimited JSON) in chunks."""
        df = pl.read_ndjson(path)
        total_rows = len(df)
        for start in range(0, total_rows, chunk_size):
            yield df.slice(start, min(chunk_size, total_rows - start))

    def transform_chunk(self, chunk: pl.DataFrame) -> list[Document]:
        """Transform a DataFrame chunk into Document objects.

        Args:
            chunk: A Polars DataFrame chunk.

        Returns:
            List of Document objects.
        """
        documents: list[Document] = []
        for row_dict in chunk.iter_rows(named=True):
            try:
                doc = self._transform_row(row_dict)
                documents.append(doc)
            except Exception:
                continue
        return documents

    @staticmethod
    def _extract_rating_number(rating_str: str) -> float | None:
        """Extract numeric rating from a string like 'Rated 1 out of 5 stars'."""
        match = re.search(r"(\d+(?:\.\d+)?)", rating_str)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        """Convert a value to string, returning None for empty/null."""
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert a value to float, returning None for invalid values."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
