"""SemanticRelationshipStore – unified Parquet store for relationships."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pain_intelligence import PIPELINE_VERSION, SCHEMA_VERSION
from pain_intelligence.knowledge.metadata import (
    make_asset_metadata,
    read_parquet_metadata,
    write_parquet_with_metadata,
)
from phase2.similarity.schema import RelationshipType, SemanticRelationship


class SemanticRelationshipStore:
    """Reads and writes semantic relationships as a single Parquet file.

    ALWAYS overwrites existing data. Contains embedded run_id metadata.
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._path = base_path / "semantic_relationships.parquet"
        self._run_id: str = ""

    @property
    def path(self) -> Path:
        return self._path

    def set_run_id(self, run_id: str) -> None:
        self._run_id = run_id

    def get_run_id(self) -> str:
        return read_parquet_metadata(self._path).get("run_id", "")

    @staticmethod
    def _record_to_row(r: SemanticRelationship) -> dict:
        return {
            "relationship_id": r.relationship_id,
            "source_type": r.source_type.value,
            "source_id": r.source_id,
            "target_type": r.target_type.value,
            "target_id": r.target_id,
            "relationship_type": r.relationship_type.value,
            "similarity_score": r.similarity_score,
            "confidence": r.confidence,
            "metric": r.metric,
            "provider": r.provider,
            "model_fingerprint": r.model_fingerprint,
            "shared_entities": r.shared_entities,
            "shared_categories": r.shared_categories,
            "support_count": r.support_count,
            "metadata": json.dumps(r.metadata),
            "version": r.version,
            "created_at": r.created_at,
        }

    @staticmethod
    def _row_to_record(row: dict) -> SemanticRelationship:
        return SemanticRelationship(
            relationship_id=row["relationship_id"],
            source_type=row["source_type"],
            source_id=row["source_id"],
            target_type=row["target_type"],
            target_id=row["target_id"],
            relationship_type=row["relationship_type"],
            similarity_score=row["similarity_score"],
            confidence=row["confidence"],
            metric=row["metric"],
            provider=row["provider"],
            model_fingerprint=row["model_fingerprint"],
            shared_entities=row.get("shared_entities") or [],
            shared_categories=row.get("shared_categories") or [],
            support_count=row.get("support_count", 0),
            metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
            version=row["version"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _schema() -> dict[str, type]:
        return {
            "relationship_id": str,
            "source_type": str,
            "source_id": str,
            "target_type": str,
            "target_id": str,
            "relationship_type": str,
            "similarity_score": float,
            "confidence": float,
            "metric": str,
            "provider": str,
            "model_fingerprint": str,
            "shared_entities": list,
            "shared_categories": list,
            "support_count": int,
            "metadata": str,
            "version": str,
            "created_at": str,
        }

    def save(self, relationships: list[SemanticRelationship]) -> Path:
        """Write relationships, ALWAYS overwriting. Writes empty parquet if no records."""
        metadata = make_asset_metadata(
            run_id=self._run_id,
            record_count=len(relationships),
        )

        if not relationships:
            df = pl.DataFrame(schema=self._schema())
        else:
            rows = [self._record_to_row(r) for r in relationships]
            df = pl.DataFrame(rows, schema=self._schema())

        return write_parquet_with_metadata(df, self._path, metadata=metadata)

    def append(self, relationships: list[SemanticRelationship]) -> Path:
        if not relationships:
            return self._path
        rows = [self._record_to_row(r) for r in relationships]
        new_df = pl.DataFrame(rows, schema=self._schema())
        if self._path.exists():
            existing = pl.read_parquet(str(self._path))
            df = pl.concat([existing, new_df], how="vertical")
        else:
            df = new_df
        metadata = make_asset_metadata(
            run_id=self._run_id,
            record_count=df.height,
        )
        return write_parquet_with_metadata(df, self._path, metadata=metadata)

    def load(self) -> list[SemanticRelationship]:
        if not self._path.exists():
            return []
        df = pl.read_parquet(str(self._path))
        return [self._row_to_record(row) for row in df.to_dicts()]

    def load_df(self) -> pl.DataFrame:
        if not self._path.exists():
            return pl.DataFrame(schema=self._schema())
        return pl.read_parquet(str(self._path))

    def count(self) -> int:
        if not self._path.exists():
            return 0
        df = pl.read_parquet(str(self._path))
        return df.height

    def exists(self) -> bool:
        return self._path.exists()
