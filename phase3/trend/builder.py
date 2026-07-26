"""TrendBuilder — end-to-end trend detection pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from phase3.trend.cache import TrendCache
from phase3.trend.config import TrendConfig
from phase3.trend.correlator import TrendCorrelator
from phase3.trend.detector import TrendDetector
from phase3.trend.scoring import TrendScorer
from phase3.trend.schema import Trend, TrendMetadata, TrendOutput
from phase3.trend.snapshot import (
    TrendSnapshotBuilder,
    TrendSnapshotComparator,
    TrendSnapshotLoader,
    TrendSnapshotScanner,
)
from phase3.trend.store import TrendStore
from phase3.trend.timeline import TrendTimelineBuilder
from phase3.trend.validator import TrendValidator


class TrendBuilder:
    """Orchestrates the full trend detection pipeline.

    1. Scan snapshots
    2. Compare latest vs prior
    3. Detect trend candidates
    4. Score candidates
    5. Correlate trends
    6. Validate
    7. Persist
    """

    def __init__(self, config: TrendConfig, run_id: str | None = None) -> None:
        self._config = config
        self._run_id = run_id or _generate_run_id()
        self._store = TrendStore(self._config.trend_dir)
        self._cache = TrendCache(self._store, config)
        self._detector = TrendDetector(config)
        self._scorer = TrendScorer(config)
        self._correlator = TrendCorrelator()
        self._validator = TrendValidator()

    @property
    def store(self) -> TrendStore:
        return self._store

    @property
    def config(self) -> TrendConfig:
        return self._config

    def build(
        self,
        knowledge_dir: Path,
        force: bool = False,
    ) -> dict:
        start = time.perf_counter()

        reasoning_checksums: dict[str, str] = {}
        kg_checksums: dict[str, str] = {}
        opportunity_checksums: dict[str, str] = {}

        # Check cache
        if not force and self._cache.is_valid(
            reasoning_checksums, kg_checksums, opportunity_checksums
        ):
            cached = self._cache.load()
            if cached:
                return self._build_result(cached.trends, cached.metadata, start, from_cache=True)

        # Scan snapshots
        scanner = TrendSnapshotScanner(knowledge_dir)
        snapshots = scanner.scan()

        if len(snapshots) < self._config.min_snapshots:
            result = self._empty_result(start)
            self._store.save_metadata(
                TrendMetadata(run_id=self._run_id, total_trends=0)
            )
            return result

        # Get snapshots for comparison
        current_snap = snapshots[-1]
        prior_idx = max(0, len(snapshots) - 1 - self._config.comparison_window)
        prior_snap = snapshots[prior_idx]

        # Build timeline
        timeline_builder = TrendTimelineBuilder(knowledge_dir)
        timeline = timeline_builder.build_with_comparisons()

        # Compare snapshots
        comparator = TrendSnapshotComparator(knowledge_dir)
        delta = comparator.compare(prior_snap, current_snap)

        # Detect trend candidates
        candidates = self._detector.detect(delta, timeline, prior_snap, current_snap)

        # Convert candidates to dicts for scorer
        candidate_dicts = self._candidates_to_dicts(candidates, current_snap, prior_snap)

        # Score candidates
        context: dict[str, Any] = {
            "knowledge_dir": str(knowledge_dir),
            "total_snapshots": len(snapshots),
            "snapshot_count": len(snapshots),
            "max_observations": max(
                *(p.metrics.get("observation_count", 0) for p in timeline.points), 1
            ),
        }
        trends = self._scorer.score(candidate_dicts, context)

        # Run correlations
        trends = self._correlator.correlate(trends)

        # Validate
        valid_snapshot_ids = {s.snapshot_id for s in snapshots}
        validation = self._validator.validate(trends, valid_snapshot_ids)

        # Build metadata
        meta = TrendMetadata(
            run_id=self._run_id,
            snapshot_count=len(snapshots),
            total_trends=len(trends),
            cache_hit=False,
            elapsed_seconds=round(time.perf_counter() - start, 4),
            first_snapshot_id=snapshots[0].snapshot_id if snapshots else "",
            last_snapshot_id=current_snap.snapshot_id,
            scoring_providers_used=self._scorer.providers_used,
        )
        self._store.save_trends(trends, self._run_id)
        self._store.save_metadata(meta)

        manifest = self._build_manifest(trends, validation, start)
        self._store.save_manifest(manifest)

        # Cache
        self._cache.save(reasoning_checksums, kg_checksums, opportunity_checksums)

        return self._build_result(trends, meta, start)

    def _candidates_to_dicts(
        self,
        candidates: list,
        current_snap: Any,
        prior_snap: Any,
    ) -> list[dict]:
        result: list[dict] = []
        for c in candidates:
            trend_type, trend_dir = self._detector.classify_trend(c, None)
            metrics = self._detector.build_metrics(c, days_elapsed=1.0)

            d = {
                "trend_id": self._detector.build_trend_id(self._run_id, c.subject_id, c.subject_label),
                "title": f"{trend_type.value.title()}: {c.subject_label}",
                "subject_id": c.subject_id,
                "subject_label": c.subject_label,
                "trend_type": trend_type.value,
                "trend_direction": trend_dir.value,
                "trend_subject": c.trend_subject.value,
                "snapshot_ids": [
                    s.snapshot_id
                    for s in [prior_snap, current_snap]
                    if s is not None
                ],
                "first_snapshot_id": prior_snap.snapshot_id if prior_snap else "",
                "last_snapshot_id": current_snap.snapshot_id if current_snap else "",
                "prior_snapshot_id": prior_snap.snapshot_id if prior_snap else "",
                "metrics": metrics.model_dump(),
                "growth_pct": c.growth_pct,
                "velocity": c.velocity,
                "momentum": metrics.momentum,
                "confidence": metrics.confidence,
                "snapshot_count": 2,
                "total_observations": int(c.current_value),
                "current_value": float(c.current_value),
                "prior_value": float(c.prior_value),
                "affected_products": c.affected_products,
                "affected_companies": c.affected_companies,
                "affected_technologies": c.affected_technologies,
                "affected_platforms": c.affected_platforms,
                "avg_frequency": metrics.avg_frequency,
                "duration_days": metrics.duration_days,
                "first_seen": metrics.first_seen,
                "last_seen": metrics.last_seen,
                "peak_value": metrics.peak_value,
                "peak_date": metrics.peak_date,
                "growth_score": 0.0,
                "velocity_score": 0.0,
                "momentum_score": 0.0,
                "confidence_score": 0.0,
                "seasonality_score": 0.0,
                "anomaly_score": 0.0,
                "cross_platform_score": 0.0,
            }
            result.append(d)
        return result

    def _build_result(
        self,
        trends: list[Trend],
        metadata: TrendMetadata | None,
        start: float,
        from_cache: bool = False,
    ) -> dict:
        total = len(trends)
        growing = sum(1 for t in trends if t.trend_type.value == "growing")
        declining = sum(1 for t in trends if t.trend_type.value == "declining")
        emerging = sum(1 for t in trends if t.trend_type.value == "emerging")
        stable = sum(1 for t in trends if t.trend_type.value == "stable")
        scores = [t.metrics.trend_score for t in trends]
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        return {
            "run_id": self._run_id,
            "total_trends": total,
            "growing": growing,
            "declining": declining,
            "emerging": emerging,
            "stable": stable,
            "avg_trend_score": avg_score,
            "cache_hit": from_cache or (metadata.cache_hit if metadata else False),
            "elapsed_seconds": round(time.perf_counter() - start, 4),
        }

    def _empty_result(self, start: float) -> dict:
        return {
            "run_id": self._run_id,
            "total_trends": 0,
            "growing": 0,
            "declining": 0,
            "emerging": 0,
            "stable": 0,
            "avg_trend_score": 0.0,
            "cache_hit": False,
            "elapsed_seconds": round(time.perf_counter() - start, 4),
        }

    def _build_manifest(self, trends: list[Trend], validation: Any, start: float) -> dict:
        return {
            "run_id": self._run_id,
            "trend_count": len(trends),
            "valid": validation.valid if validation else True,
            "validation_errors": validation.errors if validation else [],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pipeline_version": "1.0",
        }


def _generate_run_id() -> str:
    import hashlib
    raw = f"trend-{time.time_ns()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
