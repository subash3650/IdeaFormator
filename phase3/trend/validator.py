"""TrendValidator — validates trend integrity and consistency."""

from __future__ import annotations

from dataclasses import dataclass, field

from phase3.trend.schema import Trend


@dataclass
class ValidationResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trends_checked: int = 0
    duplicate_count: int = 0
    missing_snapshot_count: int = 0
    excess_growth_count: int = 0
    inconsistent_direction_count: int = 0


class TrendValidator:
    """Validates detected trends for consistency and correctness."""

    def validate(
        self,
        trends: list[Trend],
        valid_snapshot_ids: set[str] | None = None,
    ) -> ValidationResult:
        result = ValidationResult(trends_checked=len(trends))

        if not trends:
            result.warnings.append("No trends to validate")
            return result

        seen_ids: set[str] = set()
        errors: list[str] = []

        for t in trends:
            if t.trend_id in seen_ids:
                errors.append(f"Duplicate trend_id: {t.trend_id}")
                result.duplicate_count += 1
            seen_ids.add(t.trend_id)

            if valid_snapshot_ids is not None:
                for sid in t.snapshot_ids:
                    if sid not in valid_snapshot_ids:
                        result.missing_snapshot_count += 1
                        if result.missing_snapshot_count <= 5:
                            errors.append(
                                f"Trend {t.trend_id}: references unknown snapshot {sid}"
                            )

            if t.metrics.growth_pct > 50000:
                errors.append(f"Trend {t.trend_id}: growth_pct={t.metrics.growth_pct} is excessively large")
                result.excess_growth_count += 1

            if t.trend_direction.value == "up" and t.metrics.growth_pct < 0:
                errors.append(f"Trend {t.trend_id}: direction=up but growth_pct={t.metrics.growth_pct} is negative")
                result.inconsistent_direction_count += 1

            if t.trend_direction.value == "down" and t.metrics.growth_pct > 0:
                errors.append(f"Trend {t.trend_id}: direction=down but growth_pct={t.metrics.growth_pct} is positive")
                result.inconsistent_direction_count += 1

        result.errors = errors
        result.valid = len(errors) == 0
        return result
