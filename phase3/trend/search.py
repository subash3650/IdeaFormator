"""TrendSearch — query and filter detected trends."""

from __future__ import annotations

from phase3.trend.schema import Trend, TrendType


class TrendSearch:
    """Search API for filtering detected trends."""

    def __init__(self, trends: list[Trend]) -> None:
        self._trends = trends

    def find_by_id(self, trend_id: str) -> Trend | None:
        for t in self._trends:
            if t.trend_id == trend_id:
                return t
        return None

    def find_growing(self, min_score: float = 0.0) -> list[Trend]:
        return [
            t for t in self._trends
            if t.trend_type == TrendType.GROWING
            and t.metrics.trend_score >= min_score
        ]

    def find_declining(self, max_score: float = 1.0) -> list[Trend]:
        return [
            t for t in self._trends
            if t.trend_type == TrendType.DECLINING
            and t.metrics.trend_score <= max_score
        ]

    def find_emerging(self, min_growth: float = 0.0) -> list[Trend]:
        return [
            t for t in self._trends
            if t.trend_type == TrendType.EMERGING
            and t.metrics.growth_pct >= min_growth
        ]

    def find_anomalies(self) -> list[Trend]:
        return [t for t in self._trends if t.trend_type == TrendType.ANOMALY]

    def find_by_platform(self, platform: str) -> list[Trend]:
        query = platform.lower()
        return [
            t for t in self._trends
            if any(query in p.lower() for p in t.affected_platforms)
        ]

    def find_by_company(self, company: str) -> list[Trend]:
        query = company.lower()
        return [
            t for t in self._trends
            if any(query in c.lower() for c in t.affected_companies)
        ]

    def find_by_product(self, product: str) -> list[Trend]:
        query = product.lower()
        return [
            t for t in self._trends
            if any(query in p.lower() for p in t.affected_products)
        ]

    def find_by_problem(self, problem_id: str) -> list[Trend]:
        return [
            t for t in self._trends
            if t.subject_id == problem_id
            and t.trend_subject.value == "problem"
        ]

    def find_by_opportunity(self, opportunity_id: str) -> list[Trend]:
        return [
            t for t in self._trends
            if t.subject_id == opportunity_id
            and t.trend_subject.value == "opportunity"
        ]

    def find_cross_platform(self) -> list[Trend]:
        return [
            t for t in self._trends
            if len(t.affected_platforms) >= 2
        ]

    def search_text(self, query: str, top_k: int = 10) -> list[Trend]:
        if not query.strip():
            return []
        query_lower = query.lower()
        scored: list[tuple[Trend, float]] = []
        for t in self._trends:
            score = 0.0
            if query_lower in t.title.lower():
                score += 3.0
            if query_lower in t.summary.lower():
                score += 2.0
            if query_lower in t.subject_label.lower():
                score += 2.0
            for lst in [
                t.affected_products, t.affected_companies,
                t.affected_technologies, t.affected_platforms,
            ]:
                if any(query_lower in item.lower() for item in lst):
                    score += 1.0
                    break
            if score > 0:
                scored.append((t, score))
        scored.sort(key=lambda x: -x[1])
        return [t for t, _ in scored[:top_k]]
