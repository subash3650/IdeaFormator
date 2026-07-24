"""File I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def ensure_directory(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary to a JSON file with UTF-8 encoding."""
    p = Path(path)
    ensure_directory(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file and return as dictionary."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_dataframe(
    df: pl.DataFrame,
    path: str | Path,
    format: str = "parquet",
) -> None:
    """Write a Polars DataFrame to disk.

    Args:
        df: DataFrame to write.
        path: Output file path.
        format: 'parquet' or 'csv'.
    """
    p = Path(path)
    ensure_directory(p.parent)

    if format == "parquet":
        df.write_parquet(p)
    elif format == "csv":
        df.write_csv(p)
    else:
        raise ValueError(f"Unsupported output format: {format}")
