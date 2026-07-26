"""IngestionEngine — the top-level orchestrator for data collection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pain_intelligence.ingestion.adapters.base import BaseAdapter
from pain_intelligence.ingestion.clients.base import HttpClient, RateLimitedClient, RetryClient
from pain_intelligence.ingestion.clients.httpx_client import HttpxClient
from pain_intelligence.ingestion.collectors.base import BaseCollector
from pain_intelligence.ingestion.config import IngestionConfig, load_ingestion_config
from pain_intelligence.ingestion.models import CollectionResult, RawDocument, SourceType, SyncState
from pain_intelligence.ingestion.pipeline.enrich import EnrichStage
from pain_intelligence.ingestion.pipeline.fetch import FetchStage
from pain_intelligence.ingestion.pipeline.normalize import NormalizeStage
from pain_intelligence.ingestion.pipeline.persist import PersistStage
from pain_intelligence.ingestion.pipeline.transform import TransformStage
from pain_intelligence.ingestion.pipeline.validate import ValidateStage
from pain_intelligence.ingestion.registry import available_collectors, create_collector
from pain_intelligence.ingestion.state.manager import StateManager
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


def _build_client(config: IngestionConfig, source: str) -> HttpClient:
    """Build a decorated HTTP client (httpx + retry + rate limiting) for a source."""
    collector_cfg = config.collectors.get(source)
    timeout = collector_cfg.timeout if collector_cfg else 30
    retries = collector_cfg.retry_count if collector_cfg else 3
    retry_delay = collector_cfg.retry_delay if collector_cfg else 1.0
    rate_limit = collector_cfg.rate_limit if collector_cfg else 1.0

    base = HttpxClient(timeout=timeout)
    retried = RetryClient(base, max_retries=retries, base_delay=retry_delay)
    rate_limited = RateLimitedClient(retried, rate_limit=rate_limit)
    return rate_limited


class IngestionEngine:
    """Top-level orchestrator for the data ingestion pipeline.

    For each enabled collector, runs the full pipeline:
    Fetch → Transform (adapter) → Normalize → Validate → Enrich → Persist

    Also manages state and exports manifests/reports.
    """

    def __init__(self, config: str | Path | IngestionConfig = "configs/ingestion.yaml") -> None:
        if isinstance(config, (str, Path)):
            self._config = load_ingestion_config(config)
        else:
            self._config = config

        self._output_base = self._config.output_base
        self._state_manager = StateManager(self._output_base)

        # Pipeline stages (stateless, reusable)
        self._fetch_stage = FetchStage()
        self._transform_stage = TransformStage()
        self._normalize_stage = NormalizeStage()
        self._validate_stage = ValidateStage()
        self._enrich_stage = EnrichStage()
        self._persist_stage = PersistStage(self._output_base)

    def run(self, sources: list[str] | None = None, force: bool = False) -> dict[str, Any]:
        """Run ingestion for all enabled or specified sources.

        Args:
            sources: List of source names to run. If None, runs all enabled.
            force: If True, ignore existing state and re-fetch.

        Returns:
            Summary dict with per-source results.
        """
        start = time.time()

        # Import collectors to trigger registration
        import pain_intelligence.ingestion.collectors  # noqa: F401

        if sources is None:
            sources = [
                name
                for name, cfg in self._config.collectors.items()
                if cfg.enabled
            ]

        results: dict[str, Any] = {}
        for source_name in sources:
            if source_name not in self._config.collectors:
                logger.warning("Skipping unknown source: {}", source_name)
                continue

            result = self.run_collector(source_name, force=force)
            results[source_name] = result

        elapsed = time.time() - start
        return {
            "status": "completed",
            "sources": results,
            "elapsed_seconds": round(elapsed, 2),
        }

    def run_collector(self, source_name: str, force: bool = False) -> dict[str, Any]:
        """Run the full pipeline for a single collector.

        Returns a result dict with document counts and file paths.
        """
        logger.info("=== Starting ingestion for {} ===", source_name)
        start = time.time()
        collector_cfg = self._config.collectors[source_name]

        # Build HTTP client
        client = _build_client(self._config, source_name)

        try:
            # Create collector
            collector = create_collector(source_name, collector_cfg, client)

            # Get adapter
            adapter: BaseAdapter = collector.adapter_class()

            # Load state
            state = self._state_manager.load_state(SourceType(source_name))
            if force:
                state = SyncState(source=SourceType(source_name))

            # Run pipeline
            all_valid: list[RawDocument] = []
            all_invalid: list[RawDocument] = []
            pages = 0
            api_calls = 0

            for batch, page in self._fetch_stage.run(collector, state):
                pages += 1

                # Transform via adapter
                transformed = self._transform_stage.run(adapter, batch, page)

                # Normalize to RawDocument
                normalized = self._normalize_stage.run(transformed)

                # Validate
                valid, invalid = self._validate_stage.run(normalized)
                all_valid.extend(valid)
                all_invalid.extend(invalid)

            api_calls = collector._api_calls

            # Enrich
            enriched = self._enrich_stage.run(all_valid)

            # Persist (JSONL + Parquet + Manifest)
            file_paths = {}
            if enriched:
                file_paths = self._persist_stage.run(source_name, enriched)

            # Persist failed records to dead letter queue
            if all_invalid:
                self._persist_failed(source_name, all_invalid)

            # Update state
            self._state_manager.record_success(SourceType(source_name), len(enriched))

            elapsed = time.time() - start
            result = {
                "status": "completed",
                "documents_collected": len(enriched),
                "documents_invalid": len(all_invalid),
                "pages_fetched": pages,
                "api_calls": api_calls,
                "file_paths": file_paths,
                "elapsed_seconds": round(elapsed, 2),
            }

            logger.info(
                "=== Completed {}: {} docs in {:.1f}s ===",
                source_name,
                len(enriched),
                elapsed,
            )
            return result

        except Exception as e:
            logger.error("Failed to run {}: {}", source_name, e)
            self._state_manager.record_failure(SourceType(source_name), str(e))
            return {"status": "failed", "error": str(e)}
        finally:
            client.close()

    def _persist_failed(self, source_name: str, documents: list[RawDocument]) -> None:
        """Write failed/invalid documents to the dead letter queue."""
        import json

        failed_dir = self._output_base / "failed" / source_name
        failed_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        path = failed_dir / f"{ts}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for doc in documents:
                line = json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, default=str)
                f.write(line + "\n")

        logger.warning("Wrote {} failed documents to {}", len(documents), path)

    def verify(self) -> dict[str, Any]:
        """Verify the integrity of stored ingestion data."""
        results: dict[str, Any] = {}

        import pain_intelligence.ingestion.collectors  # noqa: F401

        for source_name in self._config.collectors:
            source_type = SourceType(source_name)
            state = self._state_manager.load_state(source_type)

            # Check if output files exist
            raw_dir = self._output_base / "raw" / source_name
            norm_dir = self._output_base / "normalized" / source_name

            results[source_name] = {
                "state_loaded": True,
                "last_sync": state.last_sync.isoformat() if state.last_sync else None,
                "total_collected": state.total_collected,
                "failure_count": state.failure_count,
                "raw_dir_exists": raw_dir.exists(),
                "normalized_dir_exists": norm_dir.exists(),
                "raw_files": len(list(raw_dir.glob("*.jsonl"))) if raw_dir.exists() else 0,
                "parquet_files": len(list(norm_dir.glob("*.parquet"))) if norm_dir.exists() else 0,
            }

        return results

    def stats(self) -> dict[str, Any]:
        """Return ingestion statistics."""
        return self.verify()
