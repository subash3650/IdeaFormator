from __future__ import annotations

from datetime import datetime
from typing import Any

from phase3.presentation.schema import ReportFormat, ReportOutput, ReportType
from phase3.presentation.store import PresentationStore


class PresentationSearch:
    def __init__(self, store: PresentationStore) -> None:
        self._store = store

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
        results = self._store.load_all()

        if report_type is not None:
            if isinstance(report_type, str):
                report_type = ReportType(report_type)
            results = [r for r in results if r.report_type == report_type]

        if tag is not None:
            tag_lower = tag.lower()
            results = [r for r in results if any(t.lower() == tag_lower for t in r.index_entry.tags)]

        if company is not None:
            company_lower = company.lower()
            results = [r for r in results if any(c.lower() == company_lower for c in r.index_entry.companies)]

        if technology is not None:
            tech_lower = technology.lower()
            results = [r for r in results if any(t.lower() == tech_lower for t in r.index_entry.technologies)]

        if date_from is not None:
            ref = _parse_date(date_from)
            results = [r for r in results if _parse_date(r.generated_at[:10]) >= ref]

        if date_to is not None:
            ref = _parse_date(date_to)
            results = [r for r in results if _parse_date(r.generated_at[:10]) <= ref]

        if query is not None:
            q = query.lower()
            results = [
                r
                for r in results
                if q in r.title.lower()
                or q in r.report_type.value.lower()
                or any(q in t.lower() for t in r.index_entry.tags)
                or any(q in c.lower() for c in r.index_entry.companies)
                or any(q in t.lower() for t in r.index_entry.technologies)
            ]

        results.sort(key=lambda r: r.generated_at, reverse=True)
        return results[:limit]

    def find_by_id(self, report_id: str) -> ReportOutput | None:
        for r in self._store.load_all():
            if r.report_id == report_id:
                return r
        return None

    def find_by_type(self, report_type: ReportType | str) -> list[ReportOutput]:
        return self.search(report_type=report_type)

    def find_by_tag(self, tag: str) -> list[ReportOutput]:
        return self.search(tag=tag)

    def find_by_company(self, company: str) -> list[ReportOutput]:
        return self.search(company=company)

    def find_by_technology(self, technology: str) -> list[ReportOutput]:
        return self.search(technology=technology)

    def find_recent(self, limit: int = 10) -> list[ReportOutput]:
        all_reports = self._store.load_all()
        all_reports.sort(key=lambda r: r.generated_at, reverse=True)
        return all_reports[:limit]

    def find_by_date_range(self, date_from: str, date_to: str) -> list[ReportOutput]:
        return self.search(date_from=date_from, date_to=date_to)

    def search_text(self, query: str) -> list[ReportOutput]:
        return self.search(query=query)


def _parse_date(date_str: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str[: len(fmt.replace(r"%Y", "2020").replace(r"%m", "01").replace(r"%d", "01").replace(r"%H", "00").replace(r"%M", "00").replace(r"%S", "00").replace(r"%f", "000000"))], fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        pass

    return datetime.min
