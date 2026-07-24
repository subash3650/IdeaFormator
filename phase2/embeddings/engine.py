"""EmbeddingEngine – orchestrates the full embedding pipeline."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

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
    """Heuristic concatenation of text columns for embedding."""
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
    """High-level orchestrator for the embedding pipeline."""

    def __init__(self, config: EmbeddingEngineConfig) -> None:
        self._config = config
        self._provider: EmbeddingProvider | None = None
        self._cache = EmbeddingCache(maxsize=config.cache_size)
        self._store = EmbeddingStore(config.output_dir)
        self._index: LinearIndex | None = None

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = create_provider(self._config)
        return self._provider

    @property
    def store(self) -> EmbeddingStore:
        return self._store

    def generate(self, force: bool = False) -> dict[str, Any]:
        """Run the full embedding pipeline for all configured sources."""
        self._cache.clear()
        self._index = None

        if not force:
            existing = self._store.read_all()
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
            if not source_path.exists():
                totals["by_source"][source_type.value] = {"status": "skipped", "reason": f"not found: {source_path}"}
                continue

            df = pl.read_parquet(str(source_path))
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
            batch_meta: list[tuple[str, str]] = []  # (source_id, embedding_id)

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

            if records:
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
        """Return aggregate stats over all stored embeddings."""
        df = self._store.read_all()
        return compute_stats(df)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Search stored embeddings with a text query string."""
        if self._index is None:
            df = self._store.read_all()
            if df.height == 0:
                return []
            self._index = build_index(df)
        vec = self.provider.embed_one(query)
        return self._index.search(vec, k=k)

    def verify(self) -> dict[str, Any]:
        """Verify integrity of the stored embeddings."""
        df = self._store.read_all()
        result: dict[str, Any] = {"valid": True, "checks": {}}

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
    """Extract a stable source ID from a row, preferring known columns."""
    for col in ("id", "ID", "source_id", "doc_id", "review_id", "observation_id", "signal_id"):
        if col in row and row[col] is not None:
            return str(row[col])
    return hashlib.md5(str(row).encode()).hexdigest()