from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase2.evaluation.schema import (
    ClusterEvaluation,
    DocumentEvaluation,
    EmbeddingEvaluation,
    EvidenceEvaluation,
    GlobalEvaluation,
    ObservationEvaluation,
    OpportunityEvaluation,
    ReasoningEvaluation,
    RelationshipEvaluation,
    SignalEvaluation,
    TrendEvaluation,
)


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def evaluation_to_dict(ev: GlobalEvaluation) -> dict[str, Any]:
    return _serialize(ev)


def generate_summary(ev: GlobalEvaluation) -> dict[str, Any]:
    return {
        "generated_at": ev.generated_at,
        "evaluation_version": ev.evaluation_version,
        "overall_health_score": ev.overall_health_score,
        "worst_stage": ev.worst_stage,
        "stages": {
            "documents": {
                "health_score": ev.documents.health.score,
                "total_documents": ev.documents.total_documents,
                "warnings": ev.documents.health.warnings,
            },
            "observations": {
                "health_score": ev.observations.health.score,
                "total_observations": ev.observations.total_observations,
                "warnings": ev.observations.health.warnings,
            },
            "evidence": {
                "health_score": ev.evidence.health.score,
                "total_evidence": ev.evidence.total_evidence,
                "warnings": ev.evidence.health.warnings,
            },
            "signals": {
                "health_score": ev.signals.health.score,
                "accepted_signals": ev.signals.accepted_signals,
                "warnings": ev.signals.health.warnings,
            },
            "embeddings": {
                "health_score": ev.embeddings.health.score,
                "total_vectors": ev.embeddings.total_vectors,
                "warnings": ev.embeddings.health.warnings,
            },
            "relationships": {
                "health_score": ev.relationships.health.score,
                "total_relationships": ev.relationships.total_relationships,
                "warnings": ev.relationships.health.warnings,
            },
            "clusters": {
                "health_score": ev.clusters.health.score,
                "total_clusters": ev.clusters.total_clusters,
                "warnings": ev.clusters.health.warnings,
            },
            "reasoning": {
                "health_score": ev.reasoning.health.score,
                "inference_count": ev.reasoning.inference_count,
                "root_cause_count": ev.reasoning.root_cause_count,
                "warnings": ev.reasoning.health.warnings,
            },
            "opportunities": {
                "health_score": ev.opportunities.health.score,
                "total_opportunities": ev.opportunities.total_opportunities,
                "strong_pursue_count": ev.opportunities.strong_pursue_count,
                "warnings": ev.opportunities.health.warnings,
            },
            "trends": {
                "health_score": ev.trends.health.score,
                "total_trends": ev.trends.total_trends,
                "growing_count": ev.trends.growing_count,
                "emerging_count": ev.trends.emerging_count,
                "warnings": ev.trends.health.warnings,
            },
        },
        "total_warnings": len(ev.all_warnings),
        "recommendations": ev.recommendations,
        "pipeline_timing": {
            s.stage: s.elapsed_seconds for s in ev.pipeline_timing
        },
    }


def write_reports(ev: GlobalEvaluation, output_dir: str | Path = "evaluation_reports") -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full = evaluation_to_dict(ev)
    summary = generate_summary(ev)

    report_path = output_path / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, default=str, ensure_ascii=False)

    summary_path = output_path / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    return {
        "report": report_path,
        "summary": summary_path,
    }
