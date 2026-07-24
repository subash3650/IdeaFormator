"""Loader registry with automatic platform detection.

The registry inspects column names from a dataset and automatically
selects the correct loader. No manual configuration needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

import polars as pl

from pain_intelligence.loaders.base import BaseLoader
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

_REGISTRY: list[Type[BaseLoader]] = []


def register_loader(loader_cls: Type[BaseLoader]) -> Type[BaseLoader]:
    """Register a loader class in the global registry.

    Can also be used as a decorator.
    """
    _REGISTRY.append(loader_cls)
    return loader_cls


def get_loader_for_file(
    file_path: str | Path, chunk_size: int = 50_000
) -> BaseLoader:
    """Auto-detect and return the appropriate loader for a file.

    Reads the header row, then queries each registered loader
    to find one that matches the column schema.

    Args:
        file_path: Path to the dataset file.
        chunk_size: Rows per chunk for the loader.

    Returns:
        A BaseLoader instance that can handle this file.

    Raises:
        ValueError: If no loader matches the file's column schema.
    """
    path = Path(file_path)
    columns = _read_columns(path)

    for loader_cls in _REGISTRY:
        loader = loader_cls()
        if loader._detect(columns):
            logger.info(
                "Detected platform '{}' for file '{}' (columns: {})",
                loader.platform.value,
                path.name,
                columns,
            )
            return loader

    raise ValueError(
        f"No loader found for file '{path.name}' with columns: {columns}"
    )


def _read_columns(path: Path) -> list[str]:
    """Read just the header columns from a dataset file."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pl.read_csv(path, n_rows=0, has_header=True)
        return df.columns
    elif suffix == ".parquet":
        df = pl.read_parquet(path, n_rows=0)
        return df.columns
    elif suffix in (".json", ".jsonl"):
        df = pl.read_json(path, n_rows=0) if suffix == ".json" else pl.read_ndjson(path, n_rows=0)
        return df.columns
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def get_all_loaders() -> list[Type[BaseLoader]]:
    """Return all registered loader classes."""
    return list(_REGISTRY)
