"""Report aggregation for the Knowledge Extraction Engine."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from pain_intelligence.intelligence.schema import Evidence, Observation, ProblemSignal


def build_reports(
    observations: list[Observation] | None = None,
    evidence: list[Evidence] | None = None,
    signals: list[ProblemSignal] | None = None,
    filtering_stats: dict[str, Any] | None = None,
    pipeline_version: str = "1.5.0",
) -> dict[str, Any]:
    """Aggregate all results into structured report dicts."""
    reports: dict[str, Any] = {
        "pipeline_version": pipeline_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if observations is not None:
        reports["observations"] = {
            "total": len(observations),
            "by_type": _count_by(observations, "type"),
            "by_extractor": _count_by(observations, "extractor"),
            "by_method": _count_by(observations, "method"),
        }

    if evidence is not None:
        reports["evidence"] = {
            "total": len(evidence),
            "by_category": _count_by(evidence, "category"),
            "avg_confidence": round(
                sum(e.confidence for e in evidence) / len(evidence), 4
            ) if evidence else 0.0,
            "total_observations_used": sum(e.observation_count for e in evidence),
        }

    if signals is not None:
        reports["problem_signals"] = _build_signal_report(signals)

    if filtering_stats or signals is not None:
        reports["signal_quality"] = _build_signal_quality_report(
            observations=observations,
            evidence=evidence,
            signals=signals,
            filtering_stats=filtering_stats,
        )

    return reports


def _build_signal_report(signals: list[ProblemSignal]) -> dict[str, Any]:
    """Build problem signal report section."""
    return {
        "total": len(signals),
        "by_category": _count_by(signals, "category"),
        "by_entity": _count_by(signals, "entity"),
        "avg_confidence": round(
            sum(s.confidence for s in signals) / len(signals), 4
        ) if signals else 0.0,
"top_signals": [
                {
                    "signal_key": s.signal_key,
                    "signal_text": s.signal_text,
                    "category": s.category,
                    "entity": s.entity,
                    "document_count": s.document_count,
                    "avg_rating": s.avg_rating,
                    "confidence": s.confidence,
                }
                for s in sorted(signals, key=lambda x: -x.confidence)[:20]
            ],
    }


def _build_signal_quality_report(
    observations: list[Observation] | None = None,
    evidence: list[Evidence] | None = None,
    signals: list[ProblemSignal] | None = None,
    filtering_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build signal quality evaluation report.

    Covers:
      - Pipeline counts
      - Filtering summary (entity-only, generic removed)
      - Category coverage
      - Top signals and categories
      - Unknown entities and unresolved phrases (roadmap items)
    """
    quality: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 1. Pipeline counts
    total_obs = len(observations) if observations else 0
    total_ev = len(evidence) if evidence else 0
    total_sig = len(signals) if signals else 0
    quality["pipeline_counts"] = {
        "total_observations": total_obs,
        "total_evidence": total_ev,
        "total_problem_signals": total_sig,
    }

    # 2. Filtering summary
    if filtering_stats:
        quality["filtering"] = {
            "signals_removed_entity_only": filtering_stats.get("signals_removed_entity_only", 0),
            "signals_removed_generic": filtering_stats.get("signals_removed_generic", 0),
            "signals_removed_low_documents": filtering_stats.get("signals_removed_low_documents", 0),
            "signals_removed_high_rating": filtering_stats.get("signals_removed_high_rating", 0),
            "signals_removed_low_confidence": filtering_stats.get("signals_removed_low_confidence", 0),
            "total_signals_removed": sum(
                filtering_stats.get(k, 0) for k in [
                    "signals_removed_entity_only",
                    "signals_removed_generic",
                    "signals_removed_low_documents",
                    "signals_removed_high_rating",
                    "signals_removed_low_confidence",
                ]
            ),
            "signals_removed_entity_only_details": filtering_stats.get(
                "signals_removed_entity_only_details", []
            )[:20],
            "signals_removed_generic_details": filtering_stats.get(
                "signals_removed_generic_details", []
            )[:20],
        }

    # 3. Category coverage
    if signals:
        with_category = sum(1 for s in signals if s.category)
        coverage_pct = round(with_category / len(signals) * 100, 1) if signals else 0.0
        quality["category_coverage"] = {
            "total_signals": len(signals),
            "with_category": with_category,
            "without_category": len(signals) - with_category,
            "coverage_pct": coverage_pct,
            "top_20_categories": _count_by(signals, "category"),
        }
    elif evidence:
        with_category = sum(1 for e in evidence if e.category)
        coverage_pct = round(with_category / len(evidence) * 100, 1) if evidence else 0.0
        quality["category_coverage"] = {
            "total_evidence": len(evidence),
            "with_category": with_category,
            "without_category": len(evidence) - with_category,
            "coverage_pct": coverage_pct,
            "top_20_categories": _count_by(evidence, "category"),
        }

    # 4. Top signals
    if signals:
        quality["top_20_signals"] = [
            {
                "signal_key": s.signal_key,
                "signal_text": s.signal_text,
                "category": s.category,
                "entity": s.entity,
                "document_count": s.document_count,
                "observation_count": s.observation_count,
                "avg_rating": s.avg_rating,
                "confidence": s.confidence,
            }
            for s in sorted(signals, key=lambda x: -x.confidence)[:20]
        ]

    # 5. Average confidence
    if signals:
        quality["confidence"] = {
            "average": round(sum(s.confidence for s in signals) / len(signals), 4),
            "min": round(min(s.confidence for s in signals), 4),
            "max": round(max(s.confidence for s in signals), 4),
        }

    # 6. Unknown entities (entities detected but never linked to a problem signal)
    if observations and signals:
        obs_entities = _collect_entity_observations(observations)
        signal_entities = set()
        for s in signals:
            if s.entity:
                signal_entities.add(s.entity.lower())
        unknown_entities = [
            {"entity": e, "observation_count": c}
            for e, c in sorted(obs_entities.items(), key=lambda x: -x[1])[:30]
            if e not in signal_entities
        ]
        quality["unknown_entities"] = unknown_entities if unknown_entities else []

    # 7. Top unresolved phrases (observations with no category and no canonical value)
    if observations:
        unresolved = Counter()
        for o in observations:
            if not o.category and not o.canonical_value and o.value.strip():
                unresolved[o.value.strip()[:60]] += 1
        quality["top_unresolved_phrases"] = [
            {"phrase": phrase, "count": count}
            for phrase, count in unresolved.most_common(20)
            if count > 1
        ]

    return quality


def _count_by(items: list[Any], attr: str) -> dict[str, int]:
    c: dict[str, int] = {}
    for item in items:
        val = getattr(item, attr, None)
        if val is None:
            val = "unknown"
        key = str(val.value) if hasattr(val, "value") else str(val)
        c[key] = c.get(key, 0) + 1
    return dict(sorted(c.items(), key=lambda x: -x[1]))


def _collect_entity_observations(observations: list[Observation]) -> dict[str, int]:
    """Collect all observation values that look like entity names."""
    from collections import Counter
    # Use known entity extractor patterns or simply count entity-type observations
    c: Counter[str] = Counter()
    for o in observations:
        if o.entity and o.entity.strip():
            c[o.entity] += 1
    return dict(c.most_common())