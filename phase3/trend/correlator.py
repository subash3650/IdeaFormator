"""CrossPlatformCorrelator — identifies correlations between trends and entities."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from phase3.trend.schema import CorrelationType, Trend, TrendCorrelation


class TrendCorrelator:
    """Discovers correlations between trends and other entities."""

    def correlate(
        self,
        trends: list[Trend],
        context: dict[str, Any] | None = None,
    ) -> list[Trend]:
        if not trends:
            return []

        context = context or {}
        updated: list[Trend] = []
        for t in trends:
            correlations: list[TrendCorrelation] = []
            correlations.extend(self._find_cross_platform(trends, t))
            correlations.extend(self._find_product_company_correlations(t))
            updated.append(t.model_copy(update={"correlations": correlations}))
        return updated

    def _find_cross_platform(
        self,
        all_trends: list[Trend],
        trend: Trend,
    ) -> list[TrendCorrelation]:
        results: list[TrendCorrelation] = []
        platforms = set(trend.affected_platforms)
        if len(platforms) < 2:
            return results

        for other in all_trends:
            if other.trend_id == trend.trend_id:
                continue
            overlap = platforms & set(other.affected_platforms)
            if len(overlap) >= 2:
                strength = len(overlap) / max(len(platforms), 1)
                if strength >= 0.5:
                    cid = self._correlation_id(trend.trend_id, other.trend_id, CorrelationType.CROSS_PLATFORM)
                    results.append(TrendCorrelation(
                        correlation_id=cid,
                        trend_id=trend.trend_id,
                        related_entity_id=other.trend_id,
                        correlation_type=CorrelationType.CROSS_PLATFORM,
                        correlation_strength=round(strength, 4),
                        correlation_sign="positive",
                        description=f"Cross-platform correlation via {', '.join(sorted(overlap))}",
                    ))
        return results

    def _find_product_company_correlations(self, trend: Trend) -> list[TrendCorrelation]:
        results: list[TrendCorrelation] = []

        for product in trend.affected_products:
            for company in trend.affected_companies:
                cid = self._correlation_id(trend.trend_id, f"{product}:{company}", CorrelationType.COMPANY_TREND)
                results.append(TrendCorrelation(
                    correlation_id=cid,
                    trend_id=trend.trend_id,
                    related_entity_id=f"{product}:{company}",
                    correlation_type=CorrelationType.COMPANY_TREND,
                    correlation_strength=0.5,
                    correlation_sign="positive",
                    description=f"Product {product} associated with company {company}",
                ))

        for tech in trend.affected_technologies:
            for product in trend.affected_products:
                cid = self._correlation_id(trend.trend_id, f"{tech}:{product}", CorrelationType.PROBLEM_TECHNOLOGY)
                results.append(TrendCorrelation(
                    correlation_id=cid,
                    trend_id=trend.trend_id,
                    related_entity_id=f"{tech}:{product}",
                    correlation_type=CorrelationType.PROBLEM_TECHNOLOGY,
                    correlation_strength=0.5,
                    correlation_sign="positive",
                    description=f"Technology {tech} associated with product {product}",
                ))

        return results

    @staticmethod
    def _correlation_id(trend_id: str, entity_id: str, ctype: CorrelationType) -> str:
        raw = f"{trend_id}-{entity_id}-{ctype.value}-{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
