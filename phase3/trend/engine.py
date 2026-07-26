"""TrendEngine — facade for the Trend Intelligence Engine."""

from __future__ import annotations

from pathlib import Path

from phase3.trend.builder import TrendBuilder
from phase3.trend.config import TrendConfig
from phase3.trend.snapshot import TrendSnapshotBuilder, TrendSnapshotScanner
from phase3.trend.store import TrendStore


class TrendEngine:
    """High-level facade for the Trend Intelligence Engine."""

    def __init__(self, config: TrendConfig, run_id: str | None = None) -> None:
        self._config = config
        self._run_id = run_id
        self._builder = TrendBuilder(config, run_id=run_id)

    @property
    def store(self) -> TrendStore:
        return self._builder.store

    @property
    def config(self) -> TrendConfig:
        return self._config

    def create_snapshot(self, knowledge_dir: str | Path) -> dict:
        """Create a snapshot of current pipeline assets."""
        kd = Path(knowledge_dir)
        builder = TrendSnapshotBuilder(kd)
        snap = builder.create(self._run_id or "manual")
        return {
            "snapshot_id": snap.snapshot_id,
            "run_id": snap.run_id,
            "timestamp": snap.timestamp,
            "observation_count": snap.observation_count,
            "evidence_count": snap.evidence_count,
            "signal_count": snap.signal_count,
            "opportunity_count": snap.opportunity_count,
        }

    def generate(self, knowledge_dir: str | Path, force: bool = False) -> dict:
        """Run full trend detection pipeline."""
        kd = Path(knowledge_dir)
        return self._builder.build(kd, force=force)

    def stats(self) -> dict:
        """Return statistics about detected trends."""
        trends = self._builder.store.load_trends()
        metadata = self._builder.store.load_metadata()

        if not trends:
            return {
                "total_trends": 0,
                "avg_trend_score": 0.0,
                "growing": 0,
                "declining": 0,
                "emerging": 0,
                "stable": 0,
                "anomalous": 0,
                "type_distribution": {},
                "subject_distribution": {},
            }

        type_dist: dict[str, int] = {}
        subject_dist: dict[str, int] = {}
        growing = declining = emerging = stable = anomalous = 0
        scores = []

        for t in trends:
            type_dist[t.trend_type.value] = type_dist.get(t.trend_type.value, 0) + 1
            subject_dist[t.trend_subject.value] = subject_dist.get(t.trend_subject.value, 0) + 1
            scores.append(t.metrics.trend_score)

            if t.trend_type.value == "growing":
                growing += 1
            elif t.trend_type.value == "declining":
                declining += 1
            elif t.trend_type.value == "emerging":
                emerging += 1
            elif t.trend_type.value == "stable":
                stable += 1
            elif t.trend_type.value == "anomaly":
                anomalous += 1

        return {
            "total_trends": len(trends),
            "avg_trend_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "growing": growing,
            "declining": declining,
            "emerging": emerging,
            "stable": stable,
            "anomalous": anomalous,
            "type_distribution": type_dist,
            "subject_distribution": subject_dist,
            "run_id": metadata.run_id if metadata else "",
            "cache_hit": metadata.cache_hit if metadata else False,
            "elapsed_seconds": metadata.elapsed_seconds if metadata else 0.0,
        }

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search stored trends by text."""
        from phase3.trend.search import TrendSearch

        trends = self._builder.store.load_trends()
        searcher = TrendSearch(trends)

        results = searcher.search_text(query, top_k=top_k)
        return [t.model_dump(mode="json") for t in results]

    def clear_cache(self) -> None:
        """Invalidate trend cache."""
        self._builder._cache.invalidate()
