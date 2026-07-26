"""TrendExporter — generates report outputs for detected trends."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase3.trend.schema import Trend
from phase3.trend.store import TrendStore


class TrendExporter:
    """Generates report files from stored trends."""

    def __init__(self, store: TrendStore) -> None:
        self._store = store

    def export_report(self) -> Path:
        trends = self._store.load_trends()
        metadata = self._store.load_metadata()
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata.model_dump(mode="json") if metadata else {},
            "total_trends": len(trends),
            "trends": [t.model_dump(mode="json") for t in trends],
        }
        path = self._store.trend_dir / "trend_report.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def export_statistics(self) -> Path:
        trends = self._store.load_trends()
        scores = [t.metrics.trend_score for t in trends]
        growths = [t.metrics.growth_pct for t in trends]
        type_dist: dict[str, int] = {}
        subject_dist: dict[str, int] = {}
        for t in trends:
            type_dist[t.trend_type.value] = type_dist.get(t.trend_type.value, 0) + 1
            subject_dist[t.trend_subject.value] = subject_dist.get(t.trend_subject.value, 0) + 1

        data = {
            "trend_count": len(trends),
            "avg_trend_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "avg_growth_pct": round(sum(growths) / len(growths), 4) if growths else 0.0,
            "type_distribution": type_dist,
            "subject_distribution": subject_dist,
            "score_histogram": self._build_histogram(scores, 10),
            "growth_histogram": self._build_histogram(growths, 10),
        }
        path = self._store.trend_dir / "trend_statistics.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def export_dashboard(self) -> Path:
        trends = self._store.load_trends()
        top_growing = sorted(
            [t for t in trends if t.trend_type.value == "growing"],
            key=lambda t: -t.metrics.trend_score,
        )[:10]
        top_declining = sorted(
            [t for t in trends if t.trend_type.value == "declining"],
            key=lambda t: -t.metrics.trend_score,
        )[:10]
        top_emerging = [t for t in trends if t.trend_type.value == "emerging"][:10]
        anomalies = [t for t in trends if t.trend_type.value == "anomaly"]

        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_trends": len(trends),
            "top_growing": [t.model_dump(mode="json") for t in top_growing],
            "top_declining": [t.model_dump(mode="json") for t in top_declining],
            "top_emerging": [t.model_dump(mode="json") for t in top_emerging],
            "anomalies": [t.model_dump(mode="json") for t in anomalies],
            "type_distribution": self._type_distribution(trends),
            "subject_distribution": self._subject_distribution(trends),
        }
        path = self._store.trend_dir / "trend_dashboard.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def export_dashboard_text(self) -> Path:
        trends = self._store.load_trends()
        lines = [
            "=" * 60,
            "TREND INTELLIGENCE — DASHBOARD",
            "=" * 60,
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            f"Total Trends: {len(trends)}",
            "",
        ]
        type_dist = self._type_distribution(trends)
        lines.append("─" * 40)
        lines.append("TREND TYPE DISTRIBUTION")
        lines.append("─" * 40)
        for ttype, count in sorted(type_dist.items()):
            bar = "█" * max(1, count)
            lines.append(f"  {ttype:20s} {count:4d} |{bar}")

        lines += ["", "─" * 40, "TOP GROWING", "─" * 40]
        top_growing = sorted(
            [t for t in trends if t.trend_type.value == "growing"],
            key=lambda t: -t.metrics.trend_score,
        )[:5]
        for t in top_growing:
            lines.append(f"  {t.title:40s} score={t.metrics.trend_score:.2f} growth={t.metrics.growth_pct:.1f}%")

        lines += ["", "─" * 40, "TOP DECLINING", "─" * 40]
        top_declining = sorted(
            [t for t in trends if t.trend_type.value == "declining"],
            key=lambda t: -t.metrics.trend_score,
        )[:5]
        for t in top_declining:
            lines.append(f"  {t.title:40s} score={t.metrics.trend_score:.2f} growth={t.metrics.growth_pct:.1f}%")

        lines += ["", "=" * 60]
        path = self._store.trend_dir / "trend_dashboard.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def export_summary(self) -> Path:
        trends = self._store.load_trends()
        metadata = self._store.load_metadata()
        type_dist = self._type_distribution(trends)
        subject_dist = self._subject_distribution(trends)

        lines = [
            "# Trend Intelligence Summary",
            "",
            f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
            f"**Run ID**: {metadata.run_id if metadata else 'N/A'}",
            f"**Total Trends**: {len(trends)}",
            "",
            "## Trend Type Distribution",
        ]
        for ttype, count in sorted(type_dist.items()):
            lines.append(f"- {ttype}: {count}")
        lines += ["", "## Subject Distribution"]
        for subj, count in sorted(subject_dist.items()):
            lines.append(f"- {subj}: {count}")
        lines += ["", "## Top 5 Trends"]
        top5 = sorted(trends, key=lambda t: -t.metrics.trend_score)[:5]
        for i, t in enumerate(top5, 1):
            lines.append(f"{i}. **{t.title}** ({t.trend_type.value}, score={t.metrics.trend_score:.2f})")
        lines += ["", "## Anomalies"]
        anomalies = [t for t in trends if t.trend_type.value == "anomaly"]
        if anomalies:
            for t in anomalies:
                lines.append(f"- {t.title}")
        else:
            lines.append("No anomalies detected.")
        lines.append("")

        path = self._store.trend_dir / "trend_summary.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def export_csv(self) -> Path:
        trends = self._store.load_trends()
        path = self._store.trend_dir / "trends.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trend_id", "title", "trend_type", "trend_direction",
                "subject_id", "subject_label", "growth_pct", "velocity",
                "confidence", "trend_score", "total_observations",
                "snapshot_count", "first_seen", "last_seen",
            ])
            for t in trends:
                writer.writerow([
                    t.trend_id, t.title, t.trend_type.value, t.trend_direction.value,
                    t.subject_id, t.subject_label, t.metrics.growth_pct,
                    t.metrics.velocity, t.metrics.confidence, t.metrics.trend_score,
                    t.metrics.total_observations, t.metrics.snapshot_count,
                    t.metrics.first_seen, t.metrics.last_seen,
                ])
        return path

    def export_all(self) -> dict[str, Path]:
        return {
            "report": self.export_report(),
            "statistics": self.export_statistics(),
            "dashboard": self.export_dashboard(),
            "dashboard_text": self.export_dashboard_text(),
            "summary": self.export_summary(),
            "csv": self.export_csv(),
        }

    @staticmethod
    def _type_distribution(trends: list[Trend]) -> dict[str, int]:
        dist: dict[str, int] = {}
        for t in trends:
            dist[t.trend_type.value] = dist.get(t.trend_type.value, 0) + 1
        return dist

    @staticmethod
    def _subject_distribution(trends: list[Trend]) -> dict[str, int]:
        dist: dict[str, int] = {}
        for t in trends:
            dist[t.trend_subject.value] = dist.get(t.trend_subject.value, 0) + 1
        return dist

    @staticmethod
    def _build_histogram(values: list[float], bins: int) -> list[dict[str, Any]]:
        if not values:
            return []
        mn, mx = min(values), max(values)
        if mx == mn:
            return [{"label": f"{mn:.2f}", "count": len(values), "percentage": 100.0}]
        bin_size = (mx - mn) / bins
        histogram: list[dict[str, Any]] = []
        for i in range(bins):
            lo = mn + i * bin_size
            hi = lo + bin_size
            count = sum(1 for v in values if lo <= v < hi) or (1 if i == bins - 1 else 0)
            if i == bins - 1:
                count = sum(1 for v in values if lo <= v <= hi)
            histogram.append({
                "label": f"{lo:.2f}-{hi:.2f}",
                "count": count,
                "percentage": round(count / len(values) * 100, 2),
            })
        return histogram
