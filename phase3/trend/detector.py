"""TrendDetector — identifies trend candidates from snapshot comparisons."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from phase3.trend.config import TrendConfig
from phase3.trend.growth import GrowthAnalyzer, MomentumAnalyzer, VelocityAnalyzer
from phase3.trend.schema import (
    TrendDirection,
    TrendMetrics,
    TrendSubject,
    TrendType,
)
from phase3.trend.snapshot import SnapshotDelta, TrendSnapshot
from phase3.trend.timeline import TrendTimeline, TrendTimelinePoint


class TrendCandidate:
    """A candidate trend extracted from snapshot deltas."""

    def __init__(
        self,
        subject_id: str,
        subject_label: str,
        trend_subject: TrendSubject,
        prior_value: float = 0.0,
        current_value: float = 0.0,
        growth_pct: float = 0.0,
        velocity: float = 0.0,
        prior_snapshot: TrendSnapshot | None = None,
        current_snapshot: TrendSnapshot | None = None,
        affected_products: list[str] | None = None,
        affected_companies: list[str] | None = None,
        affected_technologies: list[str] | None = None,
        affected_platforms: list[str] | None = None,
    ) -> None:
        self.subject_id = subject_id
        self.subject_label = subject_label
        self.trend_subject = trend_subject
        self.prior_value = prior_value
        self.current_value = current_value
        self.growth_pct = growth_pct
        self.velocity = velocity
        self.prior_snapshot = prior_snapshot
        self.current_snapshot = current_snapshot
        self.affected_products = affected_products or []
        self.affected_companies = affected_companies or []
        self.affected_technologies = affected_technologies or []
        self.affected_platforms = affected_platforms or []


class TrendDetector:
    """Detects trends by analyzing snapshot comparisons and computing metrics."""

    def __init__(self, config: TrendConfig) -> None:
        self._config = config
        self._growth = GrowthAnalyzer()
        self._velocity = VelocityAnalyzer()
        self._momentum = MomentumAnalyzer()

    def detect(
        self,
        delta: SnapshotDelta,
        timeline: TrendTimeline,
        prior_snapshot: TrendSnapshot,
        current_snapshot: TrendSnapshot,
    ) -> list[TrendCandidate]:
        """Detect trend candidates from a snapshot delta and timeline."""
        candidates: list[TrendCandidate] = []

        candidates.extend(self._detect_entity_trends(delta, prior_snapshot, current_snapshot))
        candidates.extend(self._detect_opportunity_trends(delta, prior_snapshot, current_snapshot))
        candidates.extend(self._detect_volume_trends(delta, timeline, prior_snapshot, current_snapshot))

        return candidates

    def _detect_entity_trends(
        self,
        delta: SnapshotDelta,
        prior: TrendSnapshot,
        current: TrendSnapshot,
    ) -> list[TrendCandidate]:
        candidates: list[TrendCandidate] = []

        all_entities = set(delta.entity_counts_before.keys()) | set(delta.entity_counts_after.keys())

        for entity_id in sorted(all_entities):
            before = delta.entity_counts_before.get(entity_id, 0)
            after = delta.entity_counts_after.get(entity_id, 0)

            if before == 0 and after == 0:
                continue

            growth_pct = 0.0
            if before > 0:
                growth_pct = ((after - before) / before) * 100.0
            elif after > 0:
                growth_pct = 100.0

            if abs(growth_pct) < self._config.min_growth_pct and after == before:
                continue

            subject = self._classify_entity(entity_id)

            candidates.append(TrendCandidate(
                subject_id=entity_id,
                subject_label=entity_id,
                trend_subject=subject,
                prior_value=float(before),
                current_value=float(after),
                growth_pct=growth_pct,
                velocity=(after - before) / 1.0,
                prior_snapshot=prior,
                current_snapshot=current,
            ))

        return candidates

    def _detect_opportunity_trends(
        self,
        delta: SnapshotDelta,
        prior: TrendSnapshot,
        current: TrendSnapshot,
    ) -> list[TrendCandidate]:
        candidates: list[TrendCandidate] = []
        before = delta.opportunity_count_before
        after = delta.opportunity_count_after

        if before > 0 or after > 0:
            growth_pct = 0.0
            if before > 0:
                growth_pct = ((after - before) / before) * 100.0
            elif after > 0:
                growth_pct = 100.0

            candidates.append(TrendCandidate(
                subject_id="opportunities",
                subject_label="Total Opportunities",
                trend_subject=TrendSubject.OPPORTUNITY,
                prior_value=float(before),
                current_value=float(after),
                growth_pct=growth_pct,
                velocity=(after - before) / 1.0,
                prior_snapshot=prior,
                current_snapshot=current,
            ))
        return candidates

    def _detect_volume_trends(
        self,
        delta: SnapshotDelta,
        timeline: TrendTimeline,
        prior: TrendSnapshot,
        current: TrendSnapshot,
    ) -> list[TrendCandidate]:
        candidates: list[TrendCandidate] = []
        volume_metrics = [
            ("observations", delta.observation_count_before, delta.observation_count_after,
             "Total Observations", TrendSubject.SIGNAL),
            ("evidence", delta.evidence_count_before, delta.evidence_count_after,
             "Total Evidence", TrendSubject.SIGNAL),
            ("signals", delta.signal_count_before, delta.signal_count_after,
             "Total Signals", TrendSubject.SIGNAL),
        ]
        for name, before, after, label, subject in volume_metrics:
            if before == 0 and after == 0:
                continue
            growth_pct = 0.0
            if before > 0:
                growth_pct = ((after - before) / before) * 100.0
            elif after > 0:
                growth_pct = 100.0
            if abs(growth_pct) < self._config.min_growth_pct and after == before:
                continue
            candidates.append(TrendCandidate(
                subject_id=f"volume_{name}",
                subject_label=label,
                trend_subject=subject,
                prior_value=float(before),
                current_value=float(after),
                growth_pct=growth_pct,
                velocity=(after - before) / 1.0,
                prior_snapshot=prior,
                current_snapshot=current,
            ))
        return candidates

    def classify_trend(
        self,
        candidate: TrendCandidate,
        timeline: TrendTimeline,
    ) -> tuple[TrendType, TrendDirection]:
        growth = candidate.growth_pct
        if growth > self._config.min_growth_pct:
            return TrendType.GROWING, TrendDirection.UP
        elif growth < -self._config.min_growth_pct:
            return TrendType.DECLINING, TrendDirection.DOWN
        elif candidate.prior_value == 0 and candidate.current_value > 0:
            return TrendType.EMERGING, TrendDirection.UP
        return TrendType.STABLE, TrendDirection.FLAT

    def build_trend_id(self, run_id: str, subject_id: str, title: str) -> str:
        raw = f"{run_id}-{subject_id}-{title}-{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def build_metrics(
        self,
        candidate: TrendCandidate,
        days_elapsed: float = 1.0,
    ) -> TrendMetrics:
        growth = self._growth.compute(
            candidate.current_value, candidate.prior_value, days_elapsed
        )
        velocity = self._velocity.compute(
            [candidate.prior_value, candidate.current_value],
            [candidate.prior_snapshot.timestamp if candidate.prior_snapshot else "",
             candidate.current_snapshot.timestamp if candidate.current_snapshot else ""],
        )
        momentum = self._momentum.compute(
            growth["growth_pct"], growth["velocity"], growth["acceleration"], 2
        )

        return TrendMetrics(
            growth_pct=growth["growth_pct"],
            velocity=growth["velocity"],
            acceleration=growth["acceleration"],
            momentum=momentum,
            confidence=self._config.min_confidence,
            duration_days=int(days_elapsed),
            first_seen=candidate.prior_snapshot.timestamp if candidate.prior_snapshot else "",
            last_seen=candidate.current_snapshot.timestamp if candidate.current_snapshot else "",
            peak_value=max(candidate.prior_value, candidate.current_value),
            peak_date=candidate.current_snapshot.timestamp if candidate.current_snapshot else "",
            avg_frequency=abs(velocity.get("avg_velocity", 0)),
            snapshot_count=2,
            trend_score=0.0,
            total_observations=int(candidate.current_value),
        )

    @staticmethod
    def _classify_entity(entity_id: str) -> TrendSubject:
        lower = entity_id.lower()
        if "product" in lower or "app" in lower:
            return TrendSubject.PRODUCT
        if "company" in lower or "corp" in lower or "inc" in lower:
            return TrendSubject.COMPANY
        if "tech" in lower or "framework" in lower or "language" in lower or "library" in lower:
            return TrendSubject.TECHNOLOGY
        if "category" in lower:
            return TrendSubject.CATEGORY
        if "feature" in lower:
            return TrendSubject.FEATURE
        return TrendSubject.PROBLEM
