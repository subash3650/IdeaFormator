"""TrendTimeline — build a timeline of snapshots for temporal analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from phase3.trend.schema import TrendSnapshot
from phase3.trend.snapshot import TrendSnapshotComparator, TrendSnapshotLoader, TrendSnapshotScanner


class TrendTimelinePoint:
    """A single point in the trend timeline with aggregated metrics."""

    def __init__(self, snapshot: TrendSnapshot, metrics: dict[str, Any] | None = None) -> None:
        self.snapshot = snapshot
        self.metrics = metrics or {}

    @property
    def timestamp(self) -> str:
        return self.snapshot.timestamp

    @property
    def run_id(self) -> str:
        return self.snapshot.run_id


class TrendTimeline:
    """Ordered collection of timeline points representing historical snapshots."""

    def __init__(self, points: list[TrendTimelinePoint]) -> None:
        self._points = sorted(points, key=lambda p: p.timestamp)

    @property
    def points(self) -> list[TrendTimelinePoint]:
        return list(self._points)

    @property
    def count(self) -> int:
        return len(self._points)

    def first(self) -> TrendTimelinePoint | None:
        return self._points[0] if self._points else None

    def last(self) -> TrendTimelinePoint | None:
        return self._points[-1] if self._points else None

    def window(self, n: int) -> TrendTimeline:
        return TrendTimeline(self._points[-n:])

    def duration_days(self) -> float:
        if len(self._points) < 2:
            return 0.0
        try:
            from datetime import datetime, timezone
            t0 = datetime.fromisoformat(self._points[0].timestamp)
            t1 = datetime.fromisoformat(self._points[-1].timestamp)
            return (t1 - t0).total_seconds() / 86400.0
        except Exception:
            return 0.0


class TrendTimelineBuilder:
    """Builds a timeline from snapshots using the scanner and loader."""

    def __init__(self, knowledge_dir: Path) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._scanner = TrendSnapshotScanner(knowledge_dir)
        self._loader = TrendSnapshotLoader(knowledge_dir)
        self._comparator = TrendSnapshotComparator(knowledge_dir)

    @property
    def scanner(self) -> TrendSnapshotScanner:
        return self._scanner

    def build(self) -> TrendTimeline:
        snapshots = self._scanner.scan()
        points: list[TrendTimelinePoint] = []
        for snap in snapshots:
            metrics: dict[str, Any] = {
                "observation_count": snap.observation_count,
                "evidence_count": snap.evidence_count,
                "signal_count": snap.signal_count,
                "opportunity_count": snap.opportunity_count,
            }
            points.append(TrendTimelinePoint(snap, metrics))
        return TrendTimeline(points)

    def build_with_comparisons(self) -> TrendTimeline:
        snapshots = self._scanner.scan()
        points: list[TrendTimelinePoint] = []
        for i, snap in enumerate(snapshots):
            metrics: dict[str, Any] = {
                "observation_count": snap.observation_count,
                "evidence_count": snap.evidence_count,
                "signal_count": snap.signal_count,
                "opportunity_count": snap.opportunity_count,
            }
            if i > 0:
                prior = snapshots[i - 1]
                delta = self._comparator.compare(prior, snap)
                metrics["observation_growth_pct"] = delta.observation_growth_pct
                metrics["evidence_growth_pct"] = delta.evidence_growth_pct
                metrics["obs_delta"] = delta.observation_count_after - delta.observation_count_before
                metrics["ev_delta"] = delta.evidence_count_after - delta.evidence_count_before
            points.append(TrendTimelinePoint(snap, metrics))
        return TrendTimeline(points)
