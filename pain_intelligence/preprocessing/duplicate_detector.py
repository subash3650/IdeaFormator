"""Duplicate detector.

Uses SHA-256 content hashing to detect exact and near-duplicate documents.
Integrates with DuckDB for memory-efficient deduplication at scale.
"""

from __future__ import annotations

import hashlib

import duckdb

from pain_intelligence.preprocessing.base import TextCleanerProtocol
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class DuplicateDetector(TextCleanerProtocol):
    """Detect duplicate documents by content hashing.

    Uses a DuckDB-backed index so dedup can scale to hundreds of millions
    of documents without loading all hashes into RAM.

    Args:
        db_path: Path to DuckDB database file. ':memory:' for in-memory.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None
        self._setup_db()

    def _setup_db(self) -> None:
        """Initialize the DuckDB dedup index."""
        self._con = duckdb.connect(self._db_path)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS seen_hashes (
                content_hash VARCHAR PRIMARY KEY,
                document_id VARCHAR,
                is_duplicate BOOLEAN DEFAULT FALSE
            )
        """)

    @property
    def name(self) -> str:
        return "duplicate_detector"

    def clean(self, text: str) -> str:
        """Return text unchanged; dedup check via is_duplicate method."""
        return text

    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA-256 hash of normalized text.

        Normalization: strip, lowercase, collapse whitespace.
        """
        import re as _re
        normalized = _re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str, document_id: str) -> bool:
        """Check if text content has been seen before.

        Args:
            text: The text content to check.
            document_id: Unique ID of the document.

        Returns:
            True if this is a duplicate, False otherwise.
        """
        content_hash = self.compute_hash(text)

        result = self._con.execute(
            "SELECT is_duplicate FROM seen_hashes WHERE content_hash = ?",
            [content_hash],
        ).fetchone()

        if result is not None:
            self._con.execute(
                "UPDATE seen_hashes SET is_duplicate = TRUE WHERE content_hash = ?",
                [content_hash],
            )
            return True

        self._con.execute(
            "INSERT INTO seen_hashes (content_hash, document_id, is_duplicate) VALUES (?, ?, FALSE)",
            [content_hash, document_id],
        )
        return False

    def get_total_seen(self) -> int:
        """Return total number of unique documents seen."""
        result = self._con.execute("SELECT COUNT(*) FROM seen_hashes").fetchone()
        return result[0] if result else 0

    def get_duplicate_count(self) -> int:
        """Return total number of duplicates found."""
        result = self._con.execute(
            "SELECT COUNT(*) FROM seen_hashes WHERE is_duplicate = TRUE"
        ).fetchone()
        return result[0] if result else 0

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._con:
            self._con.close()
            self._con = None
