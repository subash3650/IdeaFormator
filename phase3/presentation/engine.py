from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase3.presentation.builder import PresentationModelBuilder
from phase3.presentation.config import PresentationConfig
from phase3.presentation.exporter import PresentationExporter
from phase3.presentation.schema import (
    PresentationModel,
    ReportComparison,
    ReportFormat,
    ReportOutput,
    ReportType,
    TrendDelta,
)
from phase3.presentation.search import PresentationSearch
from phase3.presentation.store import PresentationStore


class PresentationEngine:
    def __init__(
        self,
        config: PresentationConfig | None = None,
        base_path: Path | str | None = None,
    ) -> None:
        self._config = config or PresentationConfig()
        if base_path is not None:
            object.__setattr__(self._config, "output_dir", Path(base_path))
        self._store = PresentationStore(self._config.output_dir)
        self._builder = PresentationModelBuilder(self._config)
        self._search = PresentationSearch(self._store)
        self._exporter = PresentationExporter(self._config)

    @property
    def store(self) -> PresentationStore:
        return self._store

    @property
    def config(self) -> PresentationConfig:
        return self._config

    @property
    def search_(self) -> PresentationSearch:
        return self._search

    def generate(
        self,
        report_type: str = "executive_summary",
        template_name: str | None = None,
        output_formats: list[ReportFormat] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        return self._builder.build(
            report_type=report_type,
            template_name=template_name,
            output_formats=output_formats,
            force=force,
        )

    def stats(self) -> dict[str, Any]:
        all_reports = self._store.load_all()
        total = len(all_reports)
        if total == 0:
            return {
                "total_reports": 0,
                "by_type": {},
                "by_format": {},
                "avg_sections": 0.0,
                "avg_charts": 0.0,
                "avg_elapsed": 0.0,
                "checksums": {},
                "earliest": None,
                "latest": None,
            }

        by_type: dict[str, int] = {}
        by_format: dict[str, int] = {}
        total_sections = 0
        total_charts = 0
        total_elapsed = 0.0
        timestamps: list[str] = []

        for r in all_reports:
            by_type[r.report_type.value] = by_type.get(r.report_type.value, 0) + 1
            for f in r.formats:
                by_format[f.value] = by_format.get(f.value, 0) + 1
            total_sections += r.sections_count
            total_charts += r.charts_count
            total_elapsed += r.elapsed_seconds
            timestamps.append(r.generated_at)

        timestamps.sort()

        return {
            "total_reports": total,
            "by_type": by_type,
            "by_format": by_format,
            "avg_sections": round(total_sections / total, 1),
            "avg_charts": round(total_charts / total, 1),
            "avg_elapsed": round(total_elapsed / total, 3),
            "checksums": self._store.checksums(),
            "earliest": timestamps[0] if timestamps else None,
            "latest": timestamps[-1] if timestamps else None,
        }

    def list_reports(self, limit: int = 50) -> list[ReportOutput]:
        all_reports = self._store.load_all()
        all_reports.sort(key=lambda r: r.generated_at, reverse=True)
        return all_reports[:limit]

    def get_report(self, report_id: str) -> PresentationModel | None:
        return self._store.load_content(report_id)

    def search(
        self,
        query: str | None = None,
        report_type: ReportType | str | None = None,
        tag: str | None = None,
        company: str | None = None,
        technology: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 20,
    ) -> list[ReportOutput]:
        return self._search.search(
            query=query,
            report_type=report_type,
            tag=tag,
            company=company,
            technology=technology,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

    def compare(self, report_a_id: str, report_b_id: str) -> dict[str, Any]:
        a = self._store.load_content(report_a_id)
        b = self._store.load_content(report_b_id)

        if a is None or b is None:
            missing = report_a_id if a is None else report_b_id
            return {"error": f"Report not found: {missing}"}

        a_opps = a.opportunities
        b_opps = b.opportunities
        new_opps = [o for o in b_opps if o not in a_opps]
        removed_opps = [o for o in a_opps if o not in b_opps]

        a_companies = set(a.companies)
        b_companies = set(b.companies)
        new_companies = list(b_companies - a_companies)

        a_products = set(a.products)
        b_products = set(b.products)
        new_products = list(b_products - a_products)

        return {
            "report_a_id": report_a_id,
            "report_b_id": report_b_id,
            "new_opportunities": new_opps,
            "removed_opportunities": removed_opps,
            "new_companies": new_companies,
            "new_products": new_products,
            "a_sections": len(a.sections),
            "b_sections": len(b.sections),
            "a_tags": a.tags,
            "b_tags": b.tags,
        }

    def export(
        self,
        report_id: str,
        output_dir: str | Path | None = None,
        formats: list[ReportFormat] | None = None,
    ) -> dict[str, str]:
        return self._exporter.export(report_id, output_dir=output_dir, formats=formats)

    def clear_cache(self) -> None:
        self._builder = PresentationModelBuilder(self._config)
