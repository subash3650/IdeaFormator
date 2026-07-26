"""
Dataset Builder — Compatibility adapter between Ingestion and Intelligence pipelines.

This module exists ONLY to bridge the gap between the ingestion framework's
RawDocument schema and the legacy Intelligence Engine's expected DataFrame schema.

The RawDocument remains the canonical data model. This adapter maps RawDocument
fields to the legacy columns that the Intelligence Engine's extractors expect,
preserving full provenance in the process.

TODO: Migrate the Intelligence Engine to consume RawDocument directly.
      This module should be deleted once that migration is complete.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

# Legacy schema expected by the Intelligence Engine extractors.
# These extractors access: id, platform, text, clean_text, title, rating,
# author, country, language, created_at, document_length, metadata, raw_record,
# source_dataset, location
LEGACY_COLUMNS = [
    "id",
    "platform",
    "source_dataset",
    "title",
    "text",
    "clean_text",
    "rating",
    "author",
    "country",
    "location",
    "language",
    "created_at",
    "metadata",
    "raw_record",
    "document_length",
]

LEGACY_SCHEMA = {col: pl.Utf8 for col in LEGACY_COLUMNS}
LEGACY_SCHEMA["rating"] = pl.Float64
LEGACY_SCHEMA["document_length"] = pl.Int64


class DatasetBuilder:
    """Compatibility adapter: merges ingestion Parquet outputs into a single
    legacy-schema dataset for the Intelligence Engine.

    This is NOT the canonical data model. RawDocument is. This adapter exists
    only to support the current Intelligence Engine until it is migrated to
    consume RawDocument directly.

    Usage:
        builder = DatasetBuilder(knowledge_base="pain_intelligence/knowledge")
        result = builder.build()
    """

    def __init__(
        self,
        knowledge_base: str | Path = "pain_intelligence/knowledge",
        output_dir: str | Path | None = None,
    ) -> None:
        self._knowledge_base = Path(knowledge_base)
        self._normalized_dir = self._knowledge_base / "normalized"
        self._output_dir = Path(output_dir) if output_dir else self._knowledge_base / "processed"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        output_path: str | Path | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run the full build pipeline: scan → map → deduplicate → write → report.

        Returns a result dict with stats and file paths.
        """
        start = datetime.now(timezone.utc)
        out_path = Path(output_path) if output_path else self._output_dir / "processed.parquet"

        # Skip if output already exists and force is not set
        if out_path.exists() and not force:
            logger.info("Dataset already exists at {}, skipping (use force=True to rebuild)", out_path)
            return {"status": "skipped", "output_path": str(out_path), "total_documents": 0, "sources": {}}

        # Scan all normalized parquet files
        parquet_files = self._scan_parquet_files()
        if not parquet_files:
            logger.warning("No normalized parquet files found in {}", self._normalized_dir)
            return {"status": "empty", "total_documents": 0, "sources": {}}

        logger.info("Found {} parquet files across sources", len(parquet_files))

        # Read and map each source
        source_dfs: dict[str, pl.DataFrame] = {}
        for source_name, path in parquet_files.items():
            df = pl.read_parquet(path)
            if df.is_empty():
                logger.debug("Skipping empty file: {}", path)
                continue
            mapped = self._map_schema(df, source_name)
            source_dfs[source_name] = mapped
            logger.info("Mapped {} documents from {}", mapped.height, source_name)

        if not source_dfs:
            return {"status": "empty", "total_documents": 0, "sources": {}}

        # Combine all sources
        combined = pl.concat(list(source_dfs.values()))

        # Deduplicate
        pre_dedup_count = combined.height
        combined = self._deduplicate(combined)
        duplicates_removed = pre_dedup_count - combined.height

        # Compute per-source stats
        stats = self._compute_stats(source_dfs, duplicates_removed, start)
        stats["status"] = "success"

        # Write output
        self._write_dataset(combined, out_path)

        # Write report
        report_path = self._output_dir / "dataset_report.json"
        self._write_report(stats, report_path)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        stats["elapsed_seconds"] = round(elapsed, 2)
        stats["output_path"] = str(out_path)
        stats["report_path"] = str(report_path)

        logger.info(
            "Dataset built: {} documents ({} duplicates removed) in {:.1f}s",
            combined.height, duplicates_removed, elapsed,
        )
        return stats

    def _scan_parquet_files(self) -> dict[str, Path]:
        """Scan normalized/ for all source parquet files.

        Returns {source_name: path} for the latest file per source.
        """
        results: dict[str, Path] = {}
        if not self._normalized_dir.exists():
            return results

        for source_dir in sorted(self._normalized_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            parquet_files = sorted(source_dir.glob("*.parquet"), reverse=True)
            if parquet_files:
                results[source_dir.name] = parquet_files[0]
        return results

    def _map_schema(self, df: pl.DataFrame, source_name: str) -> pl.DataFrame:
        """Map ingestion RawDocument columns to legacy Intelligence Engine columns.

        This is the core compatibility mapping. It preserves full provenance by:
        - Keeping all original ingestion columns in a serialized `raw_record` field
        - Storing all metadata in a serialized `metadata` field
        - Mapping semantic fields to their legacy equivalents
        """
        n = df.height

        # Build the mapped DataFrame
        mapped = pl.DataFrame()

        # id ← document_id
        mapped = mapped.with_columns(
            df["document_id"].alias("id") if "document_id" in df.columns
            else pl.Series("id", [""] * n)
        )

        # platform ← source
        mapped = mapped.with_columns(
            df["source"].alias("platform") if "source" in df.columns
            else pl.Series("platform", [source_name] * n)
        )

        # source_dataset ← source
        mapped = mapped.with_columns(
            pl.lit(source_name).alias("source_dataset")
        )

        # title ← title
        mapped = mapped.with_columns(
            df["title"].fill_null("").alias("title") if "title" in df.columns
            else pl.Series("title", [""] * n)
        )

        # text ← content
        mapped = mapped.with_columns(
            df["content"].fill_null("").alias("text") if "content" in df.columns
            else pl.Series("text", [""] * n)
        )

        # clean_text ← content (same as text for now; legacy pipeline preprocessing
        # is not replicated here — the Intelligence Engine handles its own cleaning)
        mapped = mapped.with_columns(
            pl.col("text").alias("clean_text")
        )

        # rating ← extract from metadata dict (PlayStore star_rating, etc.)
        mapped = mapped.with_columns(
            self._extract_rating(df).alias("rating")
        )

        # author ← author
        mapped = mapped.with_columns(
            df["author"].fill_null("").alias("author") if "author" in df.columns
            else pl.Series("author", [""] * n)
        )

        # country ← extract from metadata
        mapped = mapped.with_columns(
            self._extract_country(df, source_name).alias("country")
        )

        # location ← null (not available from ingestion)
        mapped = mapped.with_columns(
            pl.Series("location", [""] * n)
        )

        # language ← language
        mapped = mapped.with_columns(
            df["language"].fill_null("").alias("language") if "language" in df.columns
            else pl.Series("language", [""] * n)
        )

        # created_at ← created_at
        mapped = mapped.with_columns(
            df["created_at"].fill_null("").alias("created_at") if "created_at" in df.columns
            else pl.Series("created_at", [""] * n)
        )

        # metadata ← serialized JSON of all ingestion metadata + source info
        mapped = mapped.with_columns(
            self._serialize_metadata(df, source_name).alias("metadata")
        )

        # raw_record ← serialized JSON of original ingestion columns
        mapped = mapped.with_columns(
            self._serialize_raw_record(df).alias("raw_record")
        )

        # document_length ← len(text)
        mapped = mapped.with_columns(
            pl.col("text").str.len_bytes().cast(pl.Int64).alias("document_length")
        )

        # Ensure correct dtypes
        mapped = mapped.with_columns([
            pl.col("rating").cast(pl.Float64, strict=False),
            pl.col("document_length").cast(pl.Int64, strict=False),
        ])

        return mapped

    def _extract_rating(self, df: pl.DataFrame) -> pl.Series:
        """Extract numeric rating from metadata JSON if present."""
        if "metadata" not in df.columns:
            return pl.Series("rating", [None] * df.height)

        ratings = []
        for meta_str in df["metadata"].to_list():
            try:
                meta = json.loads(meta_str) if meta_str else {}
                # PlayStore uses star_rating, YouTube uses rating
                rating = meta.get("star_rating") or meta.get("rating")
                ratings.append(float(rating) if rating is not None else None)
            except (json.JSONDecodeError, TypeError, ValueError):
                ratings.append(None)
        return pl.Series("rating", ratings)

    def _extract_country(self, df: pl.DataFrame, source_name: str) -> pl.Series:
        """Extract country from metadata JSON if present."""
        if "metadata" not in df.columns:
            return pl.Series("country", [""] * df.height)

        countries = []
        for meta_str in df["metadata"].to_list():
            try:
                meta = json.loads(meta_str) if meta_str else {}
                country = (
                    meta.get("country")
                    or meta.get("app_country")
                    or meta.get("region")
                    or ""
                )
                countries.append(str(country))
            except (json.JSONDecodeError, TypeError):
                countries.append("")
        return pl.Series("country", countries)

    def _serialize_metadata(self, df: pl.DataFrame, source_name: str) -> pl.Series:
        """Serialize all ingestion metadata into a JSON string for provenance."""
        metas = []
        for i in range(df.height):
            meta: dict[str, Any] = {
                "source": source_name,
                "ingestion_pipeline": True,
            }
            # Merge original metadata
            if "metadata" in df.columns:
                raw_meta_str = df["metadata"][i]
                try:
                    raw_meta = json.loads(raw_meta_str) if raw_meta_str else {}
                    meta.update(raw_meta)
                except (json.JSONDecodeError, TypeError):
                    pass
            # Add source_type and tags for provenance
            if "source_type" in df.columns:
                meta["source_type"] = df["source_type"][i]
            if "tags" in df.columns:
                meta["tags"] = df["tags"][i]
            if "url" in df.columns:
                meta["url"] = df["url"][i]
            metas.append(json.dumps(meta, default=str, ensure_ascii=False))
        return pl.Series("metadata", metas)

    def _serialize_raw_record(self, df: pl.DataFrame) -> pl.Series:
        """Serialize the full original ingestion record for provenance."""
        records = []
        for i in range(df.height):
            row: dict[str, Any] = {}
            for col in df.columns:
                val = df[col][i]
                row[col] = val
            records.append(json.dumps(row, default=str, ensure_ascii=False))
        return pl.Series("raw_record", records)

    def _deduplicate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove duplicate documents by id (document_id)."""
        if df.is_empty():
            return df
        return df.unique(subset=["id"], keep="first")

    def _compute_stats(
        self,
        source_dfs: dict[str, pl.DataFrame],
        duplicates_removed: int,
        start_time: datetime,
    ) -> dict[str, Any]:
        """Compute build statistics for the report."""
        total = sum(df.height for df in source_dfs.values())
        sources = {}
        for name, df in source_dfs.items():
            created = df["created_at"].to_list() if "created_at" in df.columns else []
            valid_dates = [d for d in created if d and d != ""]
            sources[name] = {
                "documents": df.height,
                "first_seen": min(valid_dates) if valid_dates else None,
                "last_seen": max(valid_dates) if valid_dates else None,
            }

        return {
            "generated_at": start_time.isoformat(),
            "total_documents": total,
            "duplicates_removed": duplicates_removed,
            "final_documents": total - duplicates_removed,
            "sources": sources,
            "schema_version": "1.0.0",
            "adapter_version": "1.0.0",
            "adapter_note": (
                "This dataset was generated by the RawDocument→Legacy compatibility adapter. "
                "RawDocument remains the canonical data model. "
                "This adapter will be removed when the Intelligence Engine is migrated "
                "to consume RawDocument directly."
            ),
        }

    def _write_dataset(self, df: pl.DataFrame, path: Path) -> None:
        """Write the final dataset to Parquet."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        logger.info("Wrote {} documents to {}", df.height, path)

    def _write_report(self, stats: dict[str, Any], path: Path) -> None:
        """Write the dataset report JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str, ensure_ascii=False)
        logger.info("Wrote dataset report to {}", path)
