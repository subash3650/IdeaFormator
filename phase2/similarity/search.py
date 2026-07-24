"""RelationshipSearcher – search stored relationships by source_id."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phase2.similarity.schema import SemanticRelationship
from phase2.similarity.store import SemanticRelationshipStore


class RelationshipSearcher:
    """Search relationships from the store without embedding queries.

    Operates on stored source_id relationships only; never embeds arbitrary text.
    """

    def __init__(self, store: SemanticRelationshipStore) -> None:
        self._store = store

    def search_by_source(self, source_id: str, k: int = 10) -> list[SemanticRelationship]:
        """Find relationships where source_id matches."""
        df = self._store.load_df()
        if df.height == 0:
            return []
        filtered = df.filter(pl.col("source_id") == source_id)
        filtered = filtered.sort("similarity_score", descending=True).head(k)
        return [self._store._row_to_record(row) for row in filtered.to_dicts()]

    def search_by_target(self, target_id: str, k: int = 10) -> list[SemanticRelationship]:
        """Find relationships where target_id matches."""
        df = self._store.load_df()
        if df.height == 0:
            return []
        filtered = df.filter(pl.col("target_id") == target_id)
        filtered = filtered.sort("similarity_score", descending=True).head(k)
        return [self._store._row_to_record(row) for row in filtered.to_dicts()]

    def search_by_pair(
        self, source_id: str, target_id: str
    ) -> SemanticRelationship | None:
        """Find the exact relationship between a source-target pair."""
        df = self._store.load_df()
        if df.height == 0:
            return None
        filtered = df.filter(
            (pl.col("source_id") == source_id) & (pl.col("target_id") == target_id)
        )
        if filtered.height == 0:
            # Try reverse direction
            filtered = df.filter(
                (pl.col("source_id") == target_id) & (pl.col("target_id") == source_id)
            )
        if filtered.height == 0:
            return None
        return self._store._row_to_record(filtered.row(0, named=True))

    def get_related_ids(self, source_id: str, min_similarity: float = 0.0) -> list[str]:
        """Return all target IDs related to a source, ordered by similarity."""
        df = self._store.load_df()
        if df.height == 0:
            return []
        filtered = df.filter(
            (pl.col("source_id") == source_id) & (pl.col("similarity_score") >= min_similarity)
        )
        # Also include reverse direction
        reverse = df.filter(
            (pl.col("target_id") == source_id) & (pl.col("similarity_score") >= min_similarity)
        )
        ids = (
            pl.concat([
                filtered.select("target_id"),
                reverse.select("source_id").rename({"source_id": "target_id"}),
            ])
            .unique()
            .sort("target_id")
            ["target_id"]
            .to_list()
        )
        return ids
