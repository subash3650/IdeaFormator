"""DuckDB integration for scalable dedup and analytics queries."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class DuckDBStore:
    """DuckDB-backed storage for dedup indexing and analytics.

    Args:
        db_path: Path to DuckDB database file.
    """

    def __init__(self, db_path: str | Path = "outputs/pain_intelligence.duckdb") -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con: duckdb.DuckDBPyConnection | None = None
        self._setup()

    def _setup(self) -> None:
        """Initialize database tables."""
        self._con = duckdb.connect(self._db_path)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR PRIMARY KEY,
                platform VARCHAR,
                source_dataset VARCHAR,
                title VARCHAR,
                text VARCHAR,
                clean_text VARCHAR,
                rating DOUBLE,
                author VARCHAR,
                country VARCHAR,
                language VARCHAR,
                document_length INTEGER
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS removed_documents (
                document_id VARCHAR,
                platform VARCHAR,
                source_dataset VARCHAR,
                text_preview VARCHAR,
                reason VARCHAR,
                original_length INTEGER
            )
        """)

    def insert_documents(self, df: pl.DataFrame) -> int:
        """Insert processed documents into the database.

        Args:
            df: DataFrame with document columns.

        Returns:
            Number of documents inserted.
        """
        if df.is_empty():
            return 0

        try:
            self._con.execute(
                "INSERT OR REPLACE INTO documents SELECT * FROM df"
            )
            return len(df)
        except Exception as e:
            logger.warning("Error inserting documents: {}", e)
            return 0

    def insert_removed(self, removed_df: pl.DataFrame) -> int:
        """Insert removed documents for audit."""
        if removed_df.is_empty():
            return 0
        try:
            self._con.execute(
                "INSERT INTO removed_documents SELECT * FROM removed_df"
            )
            return len(removed_df)
        except Exception as e:
            logger.warning("Error inserting removed documents: {}", e)
            return 0

    def get_document_count(self) -> int:
        """Return total number of stored documents."""
        result = self._con.execute("SELECT COUNT(*) FROM documents").fetchone()
        return result[0] if result else 0

    def query(self, sql: str) -> pl.DataFrame:
        """Execute a query and return results as a Polars DataFrame."""
        return self._con.execute(sql).pl()

    def close(self) -> None:
        """Close the database connection."""
        if self._con:
            self._con.close()
            self._con = None
