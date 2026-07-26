"""TrendStore — Parquet persistence for detected trends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence.knowledge.metadata import write_parquet_with_metadata
from phase3.trend.schema import (
    Trend,
    TrendCorrelation,
    TrendDirection,
    TrendMetadata,
    TrendMetrics,
    TrendScoringBreakdown,
    TrendSnapshot,
    TrendStatus,
    TrendSubject,
    TrendType,
)

# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------

TREND_SCHEMA: dict[str, pl.DataType] = {
    "trend_id": pl.Utf8,
    "title": pl.Utf8,
    "summary": pl.Utf8,
    "trend_type": pl.Utf8,
    "trend_direction": pl.Utf8,
    "trend_subject": pl.Utf8,
    "subject_id": pl.Utf8,
    "subject_label": pl.Utf8,
    "snapshot_ids": pl.Utf8,
    "first_snapshot_id": pl.Utf8,
    "last_snapshot_id": pl.Utf8,
    "prior_snapshot_id": pl.Utf8,
    "metrics": pl.Utf8,
    "scoring": pl.Utf8,
    "correlations": pl.Utf8,
    "affected_products": pl.Utf8,
    "affected_companies": pl.Utf8,
    "affected_technologies": pl.Utf8,
    "affected_platforms": pl.Utf8,
    "affected_categories": pl.Utf8,
    "affected_features": pl.Utf8,
    "status": pl.Utf8,
    "rank": pl.Int64,
    "created_at": pl.Utf8,
    "pipeline_version": pl.Utf8,
    "schema_version": pl.Utf8,
}

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _trend_to_row(t: Trend) -> dict[str, Any]:
    return {
        "trend_id": t.trend_id,
        "title": t.title,
        "summary": t.summary,
        "trend_type": t.trend_type.value,
        "trend_direction": t.trend_direction.value,
        "trend_subject": t.trend_subject.value,
        "subject_id": t.subject_id,
        "subject_label": t.subject_label,
        "snapshot_ids": json.dumps(t.snapshot_ids),
        "first_snapshot_id": t.first_snapshot_id,
        "last_snapshot_id": t.last_snapshot_id,
        "prior_snapshot_id": t.prior_snapshot_id,
        "metrics": json.dumps(t.metrics.model_dump(mode="json")),
        "scoring": json.dumps(t.scoring.model_dump(mode="json")),
        "correlations": json.dumps([c.model_dump(mode="json") for c in t.correlations]),
        "affected_products": json.dumps(t.affected_products),
        "affected_companies": json.dumps(t.affected_companies),
        "affected_technologies": json.dumps(t.affected_technologies),
        "affected_platforms": json.dumps(t.affected_platforms),
        "affected_categories": json.dumps(t.affected_categories),
        "affected_features": json.dumps(t.affected_features),
        "status": t.status.value,
        "rank": t.rank,
        "created_at": t.created_at,
        "pipeline_version": t.pipeline_version,
        "schema_version": t.schema_version,
    }


def _row_to_trend(row: dict[str, Any]) -> Trend:
    return Trend(
        trend_id=row["trend_id"],
        title=row.get("title", ""),
        summary=row.get("summary", ""),
        trend_type=TrendType(row.get("trend_type", "stable")),
        trend_direction=TrendDirection(row.get("trend_direction", "flat")),
        trend_subject=TrendSubject(row.get("trend_subject", "problem")),
        subject_id=row.get("subject_id", ""),
        subject_label=row.get("subject_label", ""),
        snapshot_ids=json.loads(row.get("snapshot_ids") or "[]"),
        first_snapshot_id=row.get("first_snapshot_id", ""),
        last_snapshot_id=row.get("last_snapshot_id", ""),
        prior_snapshot_id=row.get("prior_snapshot_id", ""),
        metrics=TrendMetrics(**json.loads(row.get("metrics") or "{}")),
        scoring=TrendScoringBreakdown(**json.loads(row.get("scoring") or "{}")),
        correlations=[TrendCorrelation(**c) for c in json.loads(row.get("correlations") or "[]")],
        affected_products=json.loads(row.get("affected_products") or "[]"),
        affected_companies=json.loads(row.get("affected_companies") or "[]"),
        affected_technologies=json.loads(row.get("affected_technologies") or "[]"),
        affected_platforms=json.loads(row.get("affected_platforms") or "[]"),
        affected_categories=json.loads(row.get("affected_categories") or "[]"),
        affected_features=json.loads(row.get("affected_features") or "[]"),
        status=TrendStatus(row.get("status", "identified")),
        rank=int(row.get("rank", 0)),
        created_at=row.get("created_at", ""),
        pipeline_version=row.get("pipeline_version", "1.0"),
        schema_version=row.get("schema_version", "1.0"),
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TrendStore:
    """Persists trends as Parquet files.

    File layout:
        {base_path}/trend/
            trends.parquet
            trend_metadata.json
            trend_manifest.json
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)
        self._trend_dir = self._base_path / "trend"
        self._trend_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        return self._base_path

    @property
    def trend_dir(self) -> Path:
        return self._trend_dir

    @property
    def trends_path(self) -> Path:
        return self._trend_dir / "trends.parquet"

    @property
    def metadata_path(self) -> Path:
        return self._trend_dir / "trend_metadata.json"

    @property
    def manifest_path(self) -> Path:
        return self._trend_dir / "trend_manifest.json"

    def save_trends(self, trends: list[Trend], run_id: str) -> Path:
        if not trends:
            df = pl.DataFrame(schema=TREND_SCHEMA)
        else:
            rows = [_trend_to_row(t) for t in trends]
            df = pl.DataFrame(rows, schema=TREND_SCHEMA)
        metadata = {
            "run_id": run_id,
            "record_count": str(len(trends)),
            "asset": "trends.parquet",
        }
        return write_parquet_with_metadata(df, self.trends_path, metadata=metadata)

    def load_trends(self) -> list[Trend]:
        if not self.trends_path.exists():
            return []
        df = pl.read_parquet(self.trends_path)
        return [_row_to_trend(row) for row in df.iter_rows(named=True)]

    def save_metadata(self, metadata: TrendMetadata) -> Path:
        self.metadata_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
        return self.metadata_path

    def load_metadata(self) -> TrendMetadata | None:
        if not self.metadata_path.exists():
            return None
        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return TrendMetadata(**data)

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        return self.manifest_path

    def checksums(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.trends_path.exists():
            h = hashlib.sha256()
            h.update(self.trends_path.read_bytes())
            result["trends.parquet"] = h.hexdigest()[:16]
        return result
