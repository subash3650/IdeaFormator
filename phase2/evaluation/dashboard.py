from __future__ import annotations

from pathlib import Path
from typing import Any

from phase2.evaluation.reports import evaluation_to_dict, generate_summary
from phase2.evaluation.schema import GlobalEvaluation


def generate_dashboard(
    ev: GlobalEvaluation,
    output_dir: str | Path = "evaluation_reports",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dash = _build_dashboard(ev)

    json_path = output_path / "evaluation_dashboard.json"
    with open(json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(dash, f, indent=2, default=str, ensure_ascii=False)

    txt_path = output_path / "evaluation_dashboard.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_format_text_dashboard(ev))

    return {"dashboard_json": json_path, "dashboard_txt": txt_path}


def _build_dashboard(ev: GlobalEvaluation) -> dict[str, Any]:
    summary = generate_summary(ev)
    return {
        "generated_at": ev.generated_at,
        "evaluation_version": ev.evaluation_version,
        "overall_health_score": ev.overall_health_score,
        "worst_stage": ev.worst_stage,
        "stages": summary["stages"],
        "total_warnings": len(ev.all_warnings),
        "warnings": ev.all_warnings[:20],
        "recommendations": ev.recommendations,
        "pipeline_timing": summary["pipeline_timing"],
        "key_metrics": _key_metrics(ev),
    }


def _key_metrics(ev: GlobalEvaluation) -> dict[str, Any]:
    return {
        "documents": ev.documents.total_documents,
        "duplicate_rate": ev.documents.duplicate_rate,
        "metadata_completeness": ev.documents.metadata_completeness,
        "observations_per_document": (
            round(ev.observations.total_observations / max(ev.documents.total_documents, 1), 2)
            if ev.observations.total_observations else 0
        ),
        "enrichment_coverage": ev.observations.knowledge_enrichment_coverage,
        "canonicalization_rate": ev.observations.canonicalization_success_rate,
        "compression_ratio": ev.evidence.compression_ratio,
        "evidence_confidence": ev.evidence.evidence_confidence.mean,
        "accepted_signals": ev.signals.accepted_signals,
        "embedding_vectors": ev.embeddings.total_vectors,
        "zero_vector_rate": ev.embeddings.zero_vector_rate,
        "avg_similarity": ev.relationships.average_similarity,
        "relationships": ev.relationships.total_relationships,
        "clusters": ev.clusters.total_clusters,
        "avg_cluster_quality": ev.clusters.quality_distribution.mean,
        "singleton_rate": ev.clusters.singleton_rate,
        "low_quality_rate": ev.clusters.low_quality_rate,
        "reasoning_inferences": ev.reasoning.inference_count,
        "reasoning_root_causes": ev.reasoning.root_cause_count,
        "reasoning_confidence": ev.reasoning.avg_inference_confidence,
        "opportunities": ev.opportunities.total_opportunities,
        "strong_pursue_opps": ev.opportunities.strong_pursue_count,
        "avg_opp_score": ev.opportunities.avg_opportunity_score,
        "trends": ev.trends.total_trends,
        "growing_trends": ev.trends.growing_count,
        "emerging_trends": ev.trends.emerging_count,
        "avg_trend_score": ev.trends.avg_trend_score,
    }


def _format_text_dashboard(ev: GlobalEvaluation) -> str:
    lines = [
        "=" * 60,
        "PAIN INTELLIGENCE — EVALUATION DASHBOARD",
        "=" * 60,
        "",
        f"Generated: {ev.generated_at}",
        f"Evaluation Version: {ev.evaluation_version}",
        "",
        f"OVERALL HEALTH SCORE: {ev.overall_health_score}/100",
        f"Worst Stage: {ev.worst_stage}",
        "",
        "─" * 40,
        "STAGE HEALTH SCORES",
        "─" * 40,
    ]

    stages = [
        ("Documents", ev.documents.health),
        ("Observations", ev.observations.health),
        ("Evidence", ev.evidence.health),
        ("Signals", ev.signals.health),
        ("Embeddings", ev.embeddings.health),
        ("Relationships", ev.relationships.health),
        ("Clusters", ev.clusters.health),
        ("Reasoning", ev.reasoning.health),
        ("Opportunities", ev.opportunities.health),
        ("Trends", ev.trends.health),
    ]
    for name, health in stages:
        bar = "█" * max(0, min(20, int(health.score / 5)))
        lines.append(f"  {name:20s} {health.score:6.1f} |{bar:<20s}|")

    lines += [
        "",
        "─" * 40,
        "KEY METRICS",
        "─" * 40,
    ]
    km = _key_metrics(ev)
    for k, v in km.items():
        lines.append(f"  {k:30s} {v}")

    if ev.all_warnings:
        lines += [
            "",
            "─" * 40,
            f"WARNINGS ({len(ev.all_warnings)})",
            "─" * 40,
        ]
        for w in ev.all_warnings:
            lines.append(f"  ! {w}")

    if ev.recommendations:
        lines += [
            "",
            "─" * 40,
            "RECOMMENDATIONS",
            "─" * 40,
        ]
        for r in ev.recommendations:
            lines.append(f"  → {r}")

    if ev.pipeline_timing:
        lines += [
            "",
            "─" * 40,
            "EVALUATION TIMING",
            "─" * 40,
        ]
        for t in ev.pipeline_timing:
            lines.append(f"  {t.stage:20s} {t.elapsed_seconds:.4f}s")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
