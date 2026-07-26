"""EmbeddingEngine – orchestrates the full embedding pipeline."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pain_intelligence import PIPELINE_VERSION
from pain_intelligence.knowledge.exceptions import MissingAssetError, StaleAssetError
from pain_intelligence.knowledge.manifest import PipelineManifest, compute_checksum, generate_run_id
from pain_intelligence.knowledge.metadata import get_run_id_from_asset, read_parquet_metadata
from phase2.embeddings.cache import EmbeddingCache
from phase2.embeddings.config import EmbeddingEngineConfig
from phase2.embeddings.exporter import generate_quality_report, write_manifest
from phase2.embeddings.metrics import EmbeddingStats, compute_stats
from phase2.embeddings.providers.base import EmbeddingProvider
from phase2.embeddings.registry import create_provider
from phase2.embeddings.schema import (
    EmbeddingJob,
    EmbeddingManifest,
    EmbeddingRecord,
    SearchResult,
    SourceType,
)
from phase2.embeddings.search import LinearIndex, build_index
from phase2.embeddings.store import EmbeddingStore


def _make_embedding_id(source_id: str, provider: str, model: str, version: str | None) -> str:
    raw = f"{source_id}|{provider}|{model}|{version or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _text_extractor(df: pl.DataFrame) -> str:
    text_cols = [c for c in df.columns if c.lower() in {"text", "content", "body", "review_text", "description", "text_snippet"}]
    parts: list[str] = []
    for col in text_cols:
        series = df[col]
        if series.is_not_null().any():
            val = series.to_list()[0] if len(series) > 0 else ""
            if isinstance(val, str) and val.strip():
                parts.append(val)
    if not parts:
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                val = df[col].to_list()[0] if len(df[col]) > 0 else ""
                if isinstance(val, str) and val.strip():
                    parts.append(val)
    return " ".join(parts)


class EmbeddingEngine:
    """High-level orchestrator for the embedding pipeline.

    Validates input assets before processing:
      - Observations/evidence/problem_signals must exist
      - Their run_ids must match the current pipeline run
      - Checksums are recorded in the manifest
    """

    def __init__(self, config: EmbeddingEngineConfig, run_id: str | None = None) -> None:
        self._config = config
        self._run_id = run_id or generate_run_id()
        self._provider: EmbeddingProvider | None = None
        self._cache = EmbeddingCache(maxsize=config.cache_size)
        self._store = EmbeddingStore(config.output_dir)
        self._store.set_run_metadata(self._run_id)
        self._index: LinearIndex | None = None
        self._manifest = PipelineManifest(
            config.output_dir.parent.parent if config.output_dir.parent.parent.exists() else Path("pain_intelligence/knowledge"),
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = create_provider(self._config)
        return self._provider

    @property
    def store(self) -> EmbeddingStore:
        return self._store

    def _validate_input_assets(self, force: bool = False) -> dict[SourceType, pl.DataFrame]:
        """Validate and load input assets.

        Checks:
          1. Asset file exists
          2. Asset run_id matches current run (if manifest tracks it)
          3. Records asset checksum in manifest

        Returns dict of source_type -> DataFrame.
        """
        from pain_intelligence.knowledge.exceptions import MissingAssetError

        sources: dict[SourceType, pl.DataFrame] = {}
        errors: list[str] = []
        stale_errors: list[str] = []
        missing_sources: list[str] = []

        for source_type, source_path in self._config.source_paths.items():
            if not source_path.exists():
                missing_sources.append(f"{source_type.value} ({source_path})")
                continue

            # Check run_id consistency
            asset_run_id = get_run_id_from_asset(source_path)
            manifest_entry = self._manifest.get_asset(f"{source_type.value}.parquet")
            manifest_run_id = manifest_entry.get("run_id", "") if manifest_entry else ""

            if asset_run_id and manifest_run_id and asset_run_id != manifest_run_id:
                if not force:
                    stale_errors.append(
                        f"Stale asset: {source_type.value} ({source_path}) has run_id={asset_run_id}, "
                        f"manifest expects {manifest_run_id}. Use --force to override."
                    )
                    continue

            # Record input checksum
            checksum = compute_checksum(source_path)
            self._store.set_run_metadata(
                run_id=self._run_id,
                input_checksum=checksum,
            )

            df = pl.read_parquet(str(source_path))
            sources[source_type] = df

        if stale_errors and not force:
            raise StaleAssetError(
                asset_path="multiple",
                expected_run_id=self._run_id,
                actual_run_id="mismatch",
            )

        if not sources and missing_sources:
            raise MissingAssetError(missing_sources[0])

        return sources

    def generate(self, force: bool = False) -> dict[str, Any]:
        """Run the full embedding pipeline for all configured sources.

        ALWAYS overwrites previous embedding files, even if empty.
        """
        self._cache.clear()
        self._index = None

        # Validate input assets
        sources = self._validate_input_assets(force=force)

        # Pre-populate cache with existing embeddings
        if not force:
            existing = self._store.read_all()
            if existing.height > 0 and "embedding_id" in existing.columns:
                for eid in existing["embedding_id"]:
                    self._cache.add(eid)

        started_at = datetime.now(timezone.utc)
        timestamp = started_at.isoformat()

        totals: dict[str, Any] = {
            "total_input": 0,
            "total_skipped": 0,
            "total_embedded": 0,
            "total_errors": 0,
            "by_source": {},
            "elapsed_seconds": 0.0,
        }
        tick = time.perf_counter()

        for source_type, source_path in self._config.source_paths.items():
            if source_type not in sources:
                totals["by_source"][source_type.value] = {
                    "status": "skipped",
                    "reason": "input validation failed or missing",
                }
                continue

            df = sources[source_type]
            totals["total_input"] += df.height

            job = EmbeddingJob(
                source_type=source_type,
                source_path=source_path,
                output_path=self._config.output_dir,
                batch_size=self._config.batch_size,
                force=force,
            )

            records: list[EmbeddingRecord] = []
            skipped = 0
            errors = 0
            batch_texts: list[str] = []
            batch_meta: list[tuple[str, str]] = []

            for row in df.iter_rows(named=True):
                source_id = _get_source_id(row)
                eid = _make_embedding_id(
                    source_id,
                    self.provider.provider_name,
                    self.provider.model_name,
                    self.provider.model_fingerprint,
                )

                if not force and self._cache.contains(eid):
                    skipped += 1
                    continue

                text = _text_extractor(pl.DataFrame([row]))
                if not text.strip():
                    errors += 1
                    continue

                batch_texts.append(text)
                batch_meta.append((source_id, eid))

                if len(batch_texts) >= self._config.batch_size:
                    records.extend(
                        self._embed_batch(
                            batch_texts, batch_meta, source_type, timestamp,
                        )
                    )
                    batch_texts.clear()
                    batch_meta.clear()

            if batch_texts:
                records.extend(
                    self._embed_batch(
                        batch_texts, batch_meta, source_type, timestamp,
                    )
                )

            # ALWAYS write, even if empty
            self._store.write(records, source_type.value)

            totals["by_source"][source_type.value] = {
                "total": df.height,
                "embedded": len(records),
                "skipped": skipped,
                "errors": errors,
            }
            totals["total_embedded"] += len(records)
            totals["total_skipped"] += skipped
            totals["total_errors"] += errors

        totals["elapsed_seconds"] = round(time.perf_counter() - tick, 2)

        manifest = EmbeddingManifest(
            provider=self.provider.provider_name,
            model=self.provider.model_name,
            model_version=self.provider.model_fingerprint,
            dimension=self.provider.dimension,
            normalize=self._config.normalize,
            num_vectors=totals["total_embedded"],
            sources=totals["by_source"],
            created_at=timestamp,
        )
        write_manifest(manifest, self._config.output_dir)

        # Update pipeline manifest
        self._manifest.start_run()
        for source_type in self._config.source_paths:
            asset_name = f"embeddings_{source_type.value}"
            out_path = self._config.output_dir / f"embeddings_{source_type.value}.parquet"
            rc = 0
            if out_path.exists():
                try:
                    rc = len(pl.read_parquet(str(out_path)))
                except Exception:
                    pass
            self._manifest.register_asset(
                name=asset_name,
                path=out_path,
                record_count=rc,
                stage="embeddings",
            )
        self._manifest.complete_run()
        self._manifest.save()

        all_df = self._store.read_all()
        generate_quality_report(all_df, self._config.output_dir)

        return totals

    def _embed_batch(
        self,
        texts: list[str],
        meta: list[tuple[str, str]],
        source_type: SourceType,
        timestamp: str,
    ) -> list[EmbeddingRecord]:
        try:
            vecs = self.provider.embed(texts)
        except Exception:
            return []
        records: list[EmbeddingRecord] = []
        for (source_id, eid), text, vec in zip(meta, texts, vecs):
            records.append(
                EmbeddingRecord(
                    embedding_id=eid,
                    source_id=source_id,
                    source_type=source_type,
                    provider=self.provider.provider_name,
                    model=self.provider.model_name,
                    model_version=self.provider.model_fingerprint,
                    dimension=self.provider.dimension,
                    embedding=vec.tolist(),
                    text_snippet=text[:200] if self._config.store_text else None,
                    created_at=timestamp,
                )
            )
            self._cache.add(eid)
        return records

    def stats(self) -> EmbeddingStats:
        df = self._store.read_all()
        return compute_stats(df)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if self._index is None:
            df = self._store.read_all()
            if df.height == 0:
                return []
            self._index = build_index(df)
        vec = self.provider.embed_one(query)
        return self._index.search(vec, k=k)

    def verify(self) -> dict[str, Any]:
        """Verify integrity of the stored embeddings with run_id validation."""
        df = self._store.read_all()
        result: dict[str, Any] = {"valid": True, "checks": {}}

        # Check run_id consistency across source types
        run_ids = set()
        for st in ("observation", "evidence", "problem_signal"):
            meta = self._store.get_asset_metadata(st)
            rid = meta.get("run_id", "")
            if rid:
                run_ids.add(rid)
        if len(run_ids) > 1:
            result["checks"]["run_id_consistency"] = f"FAIL: multiple run_ids found: {run_ids}"
            result["valid"] = False
        elif len(run_ids) == 1:
            result["checks"]["run_id_consistency"] = f"PASS ({next(iter(run_ids))})"
        else:
            result["checks"]["run_id_consistency"] = "WARN: no run_id metadata found"

        if df.height == 0:
            result["valid"] = False
            result["checks"]["has_vectors"] = "FAIL: no vectors found"
            return result

        vecs = np.stack(df["embedding"].to_list()).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1)
        norm_ok = bool(np.allclose(norms, 1.0, atol=1e-4))
        result["checks"]["normalized"] = "PASS" if norm_ok else "WARN"
        if not norm_ok:
            result["valid"] = False

        ids = df["embedding_id"].to_list()
        unique = len(set(ids)) == len(ids)
        result["checks"]["unique_ids"] = "PASS" if unique else "FAIL"
        if not unique:
            result["valid"] = False

        result["checks"]["has_vectors"] = f"PASS ({df.height} vectors)"
        result["total_vectors"] = df.height
        result["dimension"] = int(vecs.shape[1])
        return result


def _get_source_id(row: dict[str, Any]) -> str:
    for col in ("id", "ID", "source_id", "doc_id", "review_id", "observation_id", "signal_id"):
        if col in row and row[col] is not None:
            return str(row[col])
    return hashlib.md5(str(row).encode()).hexdigest()
