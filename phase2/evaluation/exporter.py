from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase2.evaluation.reports import evaluation_to_dict, generate_summary
from phase2.evaluation.schema import GlobalEvaluation


def export_all(
    ev: GlobalEvaluation,
    output_dir: str | Path = "evaluation_reports",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    # Full report
    report = evaluation_to_dict(ev)
    rp = output_path / "evaluation_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    results["report"] = rp

    # Summary
    summary = generate_summary(ev)
    sp = output_path / "evaluation_summary.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    results["summary"] = sp

    # Dashboard JSON
    dash = _dashboard_json(ev)
    dj = output_path / "evaluation_dashboard.json"
    with open(dj, "w", encoding="utf-8") as f:
        json.dump(dash, f, indent=2, default=str, ensure_ascii=False)
    results["dashboard_json"] = dj

    # Dashboard TXT
    dt = output_path / "evaluation_dashboard.txt"
    with open(dt, "w", encoding="utf-8") as f:
        f.write(_dashboard_text(ev))
    results["dashboard_txt"] = dt

    return results


def _dashboard_json(ev: GlobalEvaluation) -> dict[str, Any]:
    return {
        "generated_at": ev.generated_at,
        "evaluation_version": ev.evaluation_version,
        "overall_health_score": ev.overall_health_score,
        "worst_stage": ev.worst_stage,
        "stages": {
            "documents": {
                "health_score": ev.documents.health.score,
                "total_documents": ev.documents.total_documents,
                "duplicate_rate": ev.documents.duplicate_rate,
                "metadata_completeness": ev.documents.metadata_completeness,
            },
            "observations": {
                "health_score": ev.observations.health.score,
                "total_observations": ev.observations.total_observations,
                "enrichment_coverage": ev.observations.knowledge_enrichment_coverage,
            },
            "evidence": {
                "health_score": ev.evidence.health.score,
                "total_evidence": ev.evidence.total_evidence,
                "compression_ratio": ev.evidence.compression_ratio,
            },
            "signals": {
                "health_score": ev.signals.health.score,
                "accepted_signals": ev.signals.accepted_signals,
                "filtered_signals": ev.signals.filtered_signals,
            },
            "embeddings": {
                "health_score": ev.embeddings.health.score,
                "total_vectors": ev.embeddings.total_vectors,
                "dimension": ev.embeddings.dimension,
                "provider": ev.embeddings.provider,
                "model": ev.embeddings.model,
            },
            "relationships": {
                "health_score": ev.relationships.health.score,
                "total_relationships": ev.relationships.total_relationships,
                "average_similarity": ev.relationships.average_similarity,
            },
            "clusters": {
                "health_score": ev.clusters.health.score,
                "total_clusters": ev.clusters.total_clusters,
                "total_members": ev.clusters.total_members,
                "singleton_rate": ev.clusters.singleton_rate,
                "low_quality_rate": ev.clusters.low_quality_rate,
            },
        },
        "warnings": ev.all_warnings[:20],
        "recommendations": ev.recommendations,
        "pipeline_timing": {s.stage: s.elapsed_seconds for s in ev.pipeline_timing},
    }


def _dashboard_text(ev: GlobalEvaluation) -> str:
    bar = lambda s: "█" * max(0, min(20, int(s / 5)))
    lines = [
        "=" * 60,
        "PAIN INTELLIGENCE — EVALUATION DASHBOARD",
        "=" * 60,
        "",
        f"Generated: {ev.generated_at}",
        f"Score: {ev.overall_health_score}/100  |  Worst: {ev.worst_stage}",
        "",
        "─" * 40,
        "STAGE HEALTH",
        "─" * 40,
    ]
    for name, h in [
        ("Documents", ev.documents.health),
        ("Observations", ev.observations.health),
        ("Evidence", ev.evidence.health),
        ("Signals", ev.signals.health),
        ("Embeddings", ev.embeddings.health),
        ("Relationships", ev.relationships.health),
        ("Clusters", ev.clusters.health),
    ]:
        lines.append(f"  {name:20s} {h.score:6.1f} |{bar(h.score):20s}|")

    lines += ["", "─" * 40, "KEY METRICS", "─" * 40]
    lines.append(f"  Documents:                {ev.documents.total_documents}")
    lines.append(f"  Duplicate Rate:           {ev.documents.duplicate_rate:.1%}")
    lines.append(f"  Observations:             {ev.observations.total_observations}")
    lines.append(f"  Evidence:                 {ev.evidence.total_evidence}")
    lines.append(f"  Signals:                  {ev.signals.accepted_signals}")
    lines.append(f"  Embeddings:               {ev.embeddings.total_vectors}")
    lines.append(f"  Relationships:            {ev.relationships.total_relationships}")
    lines.append(f"  Clusters:                 {ev.clusters.total_clusters}")

    if ev.all_warnings:
        lines += ["", "─" * 40, f"WARNINGS ({len(ev.all_warnings)})", "─" * 40]
        for w in ev.all_warnings:
            lines.append(f"  ! {w}")

    if ev.recommendations:
        lines += ["", "─" * 40, "RECOMMENDATIONS", "─" * 40]
        for r in ev.recommendations:
            lines.append(f"  → {r}")

    lines += ["", "=" * 60]
    return "\n".join(lines)
