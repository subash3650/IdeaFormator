"""TrendScorer — orchestrates provider-based scoring for trend candidates."""

from __future__ import annotations

from typing import Any

from phase3.trend.config import TrendConfig
from phase3.trend.providers.registry import (
    create_trend_score_provider,
    sorted_trend_score_providers,
)
from phase3.trend.schema import Trend, TrendScoringBreakdown


class TrendScorer:
    """Orchestrates multiple trend score providers in priority order."""

    def __init__(self, config: TrendConfig) -> None:
        self._config = config
        enabled = config.enabled_scoring_providers
        all_providers = sorted_trend_score_providers()
        self._providers_used = list(enabled if enabled else all_providers)

    @property
    def providers_used(self) -> list[str]:
        return list(self._providers_used)

    def score(
        self,
        candidates: list[dict],
        context: dict[str, Any],
    ) -> list[Trend]:
        if not candidates:
            return []

        provider_names = self._providers_used
        score_context = dict(context)
        score_context["score_weights"] = self._config.score_weights.model_dump()

        trends: list[Trend] = []
        for candidate in candidates:
            merged = TrendScoringBreakdown()
            for pname in provider_names:
                try:
                    provider = create_trend_score_provider(pname)
                    partial = provider.score(candidate, score_context)
                    merged = self._merge_scores(merged, partial)
                except Exception:
                    continue

            trend = self._build_trend(candidate, merged, score_context)
            trends.append(trend)

        return trends

    def _merge_scores(
        self, base: TrendScoringBreakdown, partial: TrendScoringBreakdown
    ) -> TrendScoringBreakdown:
        updates: dict[str, float] = {}
        for field in TrendScoringBreakdown.model_fields:
            val = getattr(partial, field)
            if val != 0.0:
                updates[field] = round(val, 4)
        if not updates:
            return base
        return TrendScoringBreakdown(**{
            **base.model_dump(),
            **updates,
        })

    def _build_trend(
        self,
        candidate: dict,
        scoring: TrendScoringBreakdown,
        context: dict,
    ) -> Trend:
        from phase3.trend.schema import (
            Trend,
            TrendDirection,
            TrendMetrics,
            TrendSubject,
            TrendType,
        )

        trend_type_str = candidate.get("trend_type", "stable")
        trend_dir_str = candidate.get("trend_direction", "flat")
        subject_str = candidate.get("trend_subject", "problem")

        try:
            trend_type = TrendType(trend_type_str)
        except ValueError:
            trend_type = TrendType.STABLE
        try:
            trend_dir = TrendDirection(trend_dir_str)
        except ValueError:
            trend_dir = TrendDirection.FLAT
        try:
            subject = TrendSubject(subject_str)
        except ValueError:
            subject = TrendSubject.PROBLEM

        raw_metrics = candidate.get("metrics", {})
        metrics = TrendMetrics(**(raw_metrics if isinstance(raw_metrics, dict) else {}))

        return Trend(
            trend_id=candidate.get("trend_id", ""),
            title=candidate.get("title", candidate.get("subject_label", "")),
            summary=candidate.get("summary", f"Trend detected for {candidate.get('subject_id', 'unknown')}"),
            trend_type=trend_type,
            trend_direction=trend_dir,
            trend_subject=subject,
            subject_id=candidate.get("subject_id", ""),
            subject_label=candidate.get("subject_label", ""),
            snapshot_ids=candidate.get("snapshot_ids", []),
            first_snapshot_id=candidate.get("first_snapshot_id", ""),
            last_snapshot_id=candidate.get("last_snapshot_id", ""),
            prior_snapshot_id=candidate.get("prior_snapshot_id", ""),
            metrics=TrendMetrics(
                growth_pct=metrics.growth_pct,
                velocity=candidate.get("velocity", metrics.velocity),
                acceleration=metrics.acceleration,
                momentum=metrics.momentum,
                confidence=metrics.confidence,
                duration_days=metrics.duration_days,
                first_seen=metrics.first_seen,
                last_seen=metrics.last_seen,
                peak_value=metrics.peak_value,
                peak_date=metrics.peak_date,
                avg_frequency=candidate.get("avg_frequency", metrics.avg_frequency),
                moving_avg=metrics.moving_avg,
                trend_score=scoring.trend_score,
                total_observations=int(candidate.get("current_value", metrics.total_observations)),
                snapshot_count=metrics.snapshot_count,
            ),
            scoring=scoring,
            affected_products=candidate.get("affected_products", []),
            affected_companies=candidate.get("affected_companies", []),
            affected_technologies=candidate.get("affected_technologies", []),
            affected_platforms=candidate.get("affected_platforms", []),
            affected_categories=candidate.get("affected_categories", []),
            affected_features=candidate.get("affected_features", []),
        )
