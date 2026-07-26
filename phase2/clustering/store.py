"""SemanticClusterStore – read and write cluster sets as Parquet and JSON."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from phase2.clustering.schema import SemanticCluster


class SemanticClusterStore:
    """Reads and writes semantic clusters as a single Parquet file."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._path = self._base_path / "semantic_clusters.parquet"

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def _record_to_row(c: SemanticCluster) -> dict:
        return {
            "cluster_id": c.cluster_id,
            "representative_id": c.representative_id,
            "member_ids": list(c.member_ids),
            "member_count": c.member_count,
            "relationship_count": c.relationship_count,
            "average_similarity": c.average_similarity,
            "density": c.density,
            "quality_score": c.quality_score,
            "cluster_type": c.cluster_type.value,
            "provider": c.provider,
            "provider_version": c.provider_version,
            "algorithm": c.algorithm,
            "metadata": json.dumps(c.metadata),
            "version": c.version,
            "created_at": c.created_at,
        }

    @staticmethod
    def _row_to_record(row: dict) -> SemanticCluster:
        return SemanticCluster(
            cluster_id=row["cluster_id"],
            representative_id=row["representative_id"],
            member_ids=tuple(row["member_ids"]),
            member_count=row["member_count"],
            relationship_count=row["relationship_count"],
            average_similarity=row["average_similarity"],
            density=row["density"],
            quality_score=row["quality_score"],
            cluster_type=row["cluster_type"],
            provider=row["provider"],
            provider_version=row["provider_version"],
            algorithm=row["algorithm"],
            metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
            version=row["version"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _schema() -> dict[str, pl.DataType]:
        return {
            "cluster_id": pl.Utf8,
            "representative_id": pl.Utf8,
            "member_ids": pl.List(pl.Utf8),
            "member_count": pl.Int64,
            "relationship_count": pl.Int64,
            "average_similarity": pl.Float64,
            "density": pl.Float64,
            "quality_score": pl.Float64,
            "cluster_type": pl.Utf8,
            "provider": pl.Utf8,
            "provider_version": pl.Utf8,
            "algorithm": pl.Utf8,
            "metadata": pl.Utf8,
            "version": pl.Utf8,
            "created_at": pl.Utf8,
        }

    def save(self, clusters: list[SemanticCluster]) -> Path:
        """Write clusters, overwriting any existing file."""
        if not clusters:
            # If empty list, write an empty DataFrame with the schema
            df = pl.DataFrame(schema=self._schema())
            df.write_parquet(str(self._path))
            return self._path

        rows = [self._record_to_row(c) for c in clusters]
        df = pl.DataFrame(rows, schema=self._schema())
        df.write_parquet(str(self._path))
        return self._path

    def load(self) -> list[SemanticCluster]:
        """Load all stored clusters."""
        if not self._path.exists():
            return []
        df = pl.read_parquet(str(self._path))
        return [self._row_to_record(row) for row in df.to_dicts()]

    def load_df(self) -> pl.DataFrame:
        """Load all stored clusters as a Polars DataFrame."""
        if not self._path.exists():
            return pl.DataFrame(schema=self._schema())
        return pl.read_parquet(str(self._path))

    def count(self) -> int:
        """Return the number of stored clusters."""
        if not self._path.exists():
            return 0
        df = pl.read_parquet(str(self._path))
        return df.height

    def exists(self) -> bool:
        """Check if the store Parquet file exists."""
        return self._path.exists()
