"""Reasoning exporter — reports, dashboards, and summaries."""

from __future__ import annotations

import json
from pathlib import Path

from phase2.reasoning.schema import InferenceOutput
from phase2.reasoning.store import ReasoningStore


class ReasoningExporter:
    def __init__(self, store: ReasoningStore) -> None:
        self._store = store

    def export_report(self) -> Path:
        inferences = self._store.load_inferences()
        chains = self._store.load_chains()
        root_causes = self._store.load_root_causes()
        evidence = self._store.load_evidence_aggregations()
        metadata = self._store.load_metadata()

        report = {
            "generated_at": metadata.created_at if metadata else "",
            "run_id": metadata.run_id if metadata else "",
            "inference_count": len(inferences),
            "chain_count": len(chains),
            "root_cause_count": len(root_causes),
            "evidence_aggregation_count": len(evidence),
            "inferences_by_type": _count_by_key(inferences, "inference_type", "inference_type"),
            "chains": [c.model_dump(mode="json") for c in chains[:10]],
            "root_causes": [rc.model_dump(mode="json") for rc in root_causes[:20]],
            "evidence_aggregations": [ea.model_dump(mode="json") for ea in evidence[:20]],
            "metadata": metadata.model_dump(mode="json") if metadata else {},
            "checksums": self._store.checksums(),
        }
        path = self._store.reasoning_dir / "reasoning_report.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return path

    def export_statistics(self) -> Path:
        inferences = self._store.load_inferences()
        chains = self._store.load_chains()
        root_causes = self._store.load_root_causes()
        evidence = self._store.load_evidence_aggregations()

        type_dist = _count_by_key(inferences, "inference_type", "inference_type")
        avg_conf = 0.0
        if inferences:
            avg_conf = sum(i.confidence for i in inferences) / len(inferences)
        avg_depth = 0.0
        if root_causes:
            avg_depth = sum(rc.path_length for rc in root_causes) / len(root_causes)

        stats = {
            "inference_count": len(inferences),
            "chain_count": len(chains),
            "root_cause_count": len(root_causes),
            "evidence_aggregation_count": len(evidence),
            "inference_type_distribution": type_dist,
            "average_confidence": round(avg_conf, 4),
            "average_root_cause_depth": round(avg_depth, 4),
            "max_inference_confidence": max((i.confidence for i in inferences), default=0.0),
            "min_inference_confidence": min((i.confidence for i in inferences), default=0.0),
        }
        path = self._store.reasoning_dir / "reasoning_statistics.json"
        path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
        return path

    def export_dashboard(self) -> Path:
        inferences = self._store.load_inferences()
        type_dist = _count_by_key(inferences, "inference_type", "inference_type")
        metadata = self._store.load_metadata()

        dash = {
            "inference_count": len(inferences),
            "inferences_by_type": type_dist,
            "rules_applied": metadata.rules_applied if metadata else [],
            "rule_firing_counts": metadata.rule_firing_counts if metadata else {},
            "elapsed_seconds": metadata.elapsed_seconds if metadata else 0.0,
            "cache_hit": metadata.cache_hit if metadata else False,
        }
        path = self._store.reasoning_dir / "reasoning_dashboard.json"
        path.write_text(json.dumps(dash, indent=2, default=str), encoding="utf-8")
        return path

    def export_summary(self) -> Path:
        inferences = self._store.load_inferences()
        root_causes = self._store.load_root_causes()
        metadata = self._store.load_metadata()
        type_dist = _count_by_key(inferences, "inference_type", "inference_type")

        lines = [
            "# Reasoning Summary",
            "",
            f"**Run ID:** {metadata.run_id if metadata else 'N/A'}",
            f"**Generated:** {metadata.created_at if metadata else 'N/A'}",
            "",
            "## Overview",
            "",
            f"- Total Inferences: {len(inferences)}",
            f"- Root Causes Found: {len(root_causes)}",
            f"- Reasoning Chains: {metadata.chain_count if metadata else 0}",
            "",
            "## Inferences by Type",
            "",
            "| Type | Count |",
            "|------|-------|",
        ]
        for t, c in sorted(type_dist.items()):
            lines.append(f"| {t} | {c} |")
        if root_causes:
            lines.extend([
                "",
                "## Top Root Causes",
                "",
                "| Rank | Cause | Effect | Impact | Confidence |",
                "|------|-------|--------|--------|------------|",
            ])
            for i, rc in enumerate(root_causes[:10], 1):
                lines.append(
                    f"| {i} | {rc.cause_label[:40]} | {rc.effect_label[:40]} "
                    f"| {rc.transitive_impact_count} | {rc.propagated_confidence:.2f} |"
                )
        lines.extend([
            "",
            f"_Reasoning v1.0, elapsed {metadata.elapsed_seconds if metadata else 0:.2f}s_",
        ])

        path = self._store.reasoning_dir / "reasoning_summary.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def _count_by_key(items: list, attr: str, key_attr: str | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if hasattr(item, attr):
            val = getattr(item, attr)
            if hasattr(val, "value"):
                val = val.value
            counts[str(val)] = counts.get(str(val), 0) + 1
        elif isinstance(item, dict):
            val = item.get(attr, "unknown")
            counts[str(val)] = counts.get(str(val), 0) + 1
    return counts
