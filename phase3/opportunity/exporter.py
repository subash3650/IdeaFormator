"""OpportunityExporter — reports, dashboards, and summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from phase3.opportunity.schema import Opportunity
from phase3.opportunity.store import OpportunityStore


class OpportunityExporter:
    """Generates reports, dashboards, summaries, and CSV exports."""

    def __init__(self, store: OpportunityStore) -> None:
        self._store = store

    def export_report(self) -> Path:
        opportunities = self._store.load_opportunities()
        metadata = self._store.load_metadata()

        report = {
            "generated_at": metadata.created_at if metadata else "",
            "run_id": metadata.run_id if metadata else "",
            "total_opportunities": len(opportunities),
            "avg_opportunity_score": metadata.avg_opportunity_score if metadata else 0.0,
            "recommendation_distribution": metadata.recommendation_distribution if metadata else {},
            "business_model_distribution": metadata.business_model_distribution if metadata else {},
            "top_opportunities": [
                o.model_dump(mode="json") for o in sorted(
                    opportunities, key=lambda x: -x.opportunity_score
                )[:20]
            ],
            "opportunities_by_type": _count_by_key(opportunities, "suggested_business_model"),
            "opportunities_by_recommendation": _count_by_key(opportunities, "recommendation_type"),
            "score_distribution": _score_distribution(opportunities),
            "metadata": metadata.model_dump(mode="json") if metadata else {},
            "checksums": self._store.checksums(),
        }
        path = self._store.opportunity_dir / "opportunity_report.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return path

    def export_statistics(self) -> Path:
        opportunities = self._store.load_opportunities()
        scores = [o.opportunity_score for o in opportunities] if opportunities else [0.0]
        severities = [o.pain_severity for o in opportunities] if opportunities else [0.0]

        def _stats(vals: list[float]) -> dict:
            if not vals:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            return {
                "mean": round(sum(sorted_vals) / n, 4),
                "min": round(sorted_vals[0], 4),
                "max": round(sorted_vals[-1], 4),
                "median": round(sorted_vals[n // 2], 4),
            }

        stats = {
            "opportunity_count": len(opportunities),
            "opportunity_score": _stats(scores),
            "pain_severity": _stats(severities),
            "recommendation_distribution": _count_by_key(opportunities, "recommendation_type"),
            "business_model_distribution": _count_by_key(opportunities, "suggested_business_model"),
        }
        path = self._store.opportunity_dir / "opportunity_statistics.json"
        path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
        return path

    def export_dashboard(self) -> Path:
        opportunities = self._store.load_opportunities()
        metadata = self._store.load_metadata()

        dash = {
            "generated_at": metadata.created_at if metadata else "",
            "evaluation_version": metadata.schema_version if metadata else "1.0",
            "total_opportunities": len(opportunities),
            "avg_score": metadata.avg_opportunity_score if metadata else 0.0,
            "recommendation_distribution": metadata.recommendation_distribution if metadata else {},
            "business_model_distribution": metadata.business_model_distribution if metadata else {},
            "elapsed_seconds": metadata.elapsed_seconds if metadata else 0.0,
            "cache_hit": metadata.cache_hit if metadata else False,
            "top_opportunities": [
                {
                    "rank": o.rank,
                    "title": o.title[:80],
                    "score": o.opportunity_score,
                    "recommendation": o.recommendation_type.value,
                    "business_model": o.suggested_business_model.value,
                }
                for o in sorted(opportunities, key=lambda x: -x.opportunity_score)[:10]
            ],
            "key_metrics": _key_metrics(opportunities),
        }
        path = self._store.opportunity_dir / "opportunity_dashboard.json"
        path.write_text(json.dumps(dash, indent=2, default=str), encoding="utf-8")
        return path

    def export_dashboard_text(self) -> Path:
        opportunities = self._store.load_opportunities()
        metadata = self._store.load_metadata()

        lines = [
            "=" * 60,
            "OPPORTUNITY DISCOVERY — DASHBOARD",
            "=" * 60,
            "",
            f"Generated: {metadata.created_at if metadata else 'N/A'}",
            f"Run ID: {metadata.run_id if metadata else 'N/A'}",
            "",
            "─" * 40,
            "OVERVIEW",
            "─" * 40,
            f"  Total Opportunities:    {len(opportunities)}",
            f"  Average Score:          {metadata.avg_opportunity_score if metadata else 0:.4f}",
            f"  Elapsed:                {metadata.elapsed_seconds if metadata else 0:.2f}s",
            f"  Cache Hit:              {metadata.cache_hit if metadata else False}",
            "",
            "─" * 40,
            "RECOMMENDATION BREAKDOWN",
            "─" * 40,
        ]
        for rec, cnt in sorted((metadata.recommendation_distribution if metadata else {}).items()):
            lines.append(f"  {rec:25s} {cnt}")
        lines.extend([
            "",
            "─" * 40,
            "BUSINESS MODEL BREAKDOWN",
            "─" * 40,
        ])
        for bm, cnt in sorted((metadata.business_model_distribution if metadata else {}).items()):
            lines.append(f"  {bm:25s} {cnt}")
        lines.extend([
            "",
            "─" * 40,
            "TOP 10 OPPORTUNITIES",
            "─" * 40,
        ])
        for o in sorted(opportunities, key=lambda x: -x.opportunity_score)[:10]:
            lines.append(
                f"  #{o.rank:3d} [{o.opportunity_score:.2f}] "
                f"{o.recommendation_type.value:20s} "
                f"{o.suggested_business_model.value:20s} "
                f"{o.title[:50]}"
            )
        lines.extend([
            "",
            "=" * 60,
        ])
        path = self._store.opportunity_dir / "opportunity_dashboard.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def export_summary(self) -> Path:
        opportunities = self._store.load_opportunities()
        metadata = self._store.load_metadata()

        rec_dist = metadata.recommendation_distribution if metadata else {}
        bm_dist = metadata.business_model_distribution if metadata else {}

        lines = [
            "# Opportunity Discovery Summary",
            "",
            f"**Run ID:** {metadata.run_id if metadata else 'N/A'}",
            f"**Generated:** {metadata.created_at if metadata else 'N/A'}",
            "",
            "## Overview",
            "",
            f"- Total Opportunities: {len(opportunities)}",
            f"- Strong Pursue: {rec_dist.get('strong_pursue', 0)}",
            f"- Worth Exploring: {rec_dist.get('worth_exploring', 0)}",
            f"- Average Score: {metadata.avg_opportunity_score if metadata else 0:.4f}",
            "",
            "## Top 10 Opportunities",
            "",
            "| Rank | Title | Score | Recommendation | Business Model |",
            "|------|-------|-------|----------------|----------------|",
        ]
        for o in sorted(opportunities, key=lambda x: -x.opportunity_score)[:10]:
            lines.append(
                f"| {o.rank} | {o.title[:50]} | {o.opportunity_score:.2f} "
                f"| {o.recommendation_type.value} | {o.suggested_business_model.value} |"
            )
        lines.extend([
            "",
            "## Recommendation Distribution",
            "",
            "| Type | Count |",
            "|------|-------|",
        ])
        for k, v in sorted(rec_dist.items()):
            lines.append(f"| {k} | {v} |")
        lines.extend([
            "",
            "## Business Model Distribution",
            "",
            "| Model | Count |",
            "|-------|-------|",
        ])
        for k, v in sorted(bm_dist.items()):
            lines.append(f"| {k} | {v} |")
        lines.extend([
            "",
            f"_Opportunity Discovery v{metadata.pipeline_version if metadata else '1.0'}, "
            f"elapsed {metadata.elapsed_seconds if metadata else 0:.2f}s_",
        ])
        path = self._store.opportunity_dir / "opportunity_summary.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def export_csv(self) -> Path:
        opportunities = self._store.load_opportunities()
        path = self._store.opportunity_dir / "opportunities.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "opportunity_id", "title", "opportunity_score",
                "recommendation_type", "suggested_business_model",
                "pain_severity", "confidence", "product_count",
            ])
            for o in sorted(opportunities, key=lambda x: -x.opportunity_score):
                writer.writerow([
                    o.rank, o.opportunity_id, o.title, o.opportunity_score,
                    o.recommendation_type.value, o.suggested_business_model.value,
                    o.pain_severity, o.confidence.final_confidence,
                    len(o.affected_products),
                ])
        return path


def _count_by_key(items: list, attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if hasattr(item, attr):
            val = getattr(item, attr)
            if hasattr(val, "value"):
                val = val.value
            key = str(val)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _score_distribution(opportunities: list[Opportunity]) -> dict[str, int]:
    bins: dict[str, int] = {
        "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
        "0.6-0.8": 0, "0.8-1.0": 0,
    }
    for o in opportunities:
        s = o.opportunity_score
        if s < 0.2:
            bins["0.0-0.2"] += 1
        elif s < 0.4:
            bins["0.2-0.4"] += 1
        elif s < 0.6:
            bins["0.4-0.6"] += 1
        elif s < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    return bins


def _key_metrics(opportunities: list[Opportunity]) -> dict:
    scores = [o.opportunity_score for o in opportunities] if opportunities else [0.0]
    return {
        "total_opportunities": len(opportunities),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "strong_pursue_count": sum(
            1 for o in opportunities
            if o.recommendation_type.value == "strong_pursue"
        ),
        "avg_pain_severity": round(
            sum(o.pain_severity for o in opportunities) / max(len(opportunities), 1), 4
        ),
    }
