"""Trend stage evaluator for the pipeline evaluation framework."""

from __future__ import annotations

from pathlib import Path

from phase2.evaluation.schema import StageHealth, TrendEvaluation
from phase3.trend.store import TrendStore


class TrendEvaluator:
    """Evaluates the trend intelligence stage."""

    def __init__(self, knowledge_dir: str | Path) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._store = TrendStore(self._knowledge_dir / "assets" / "phase3")

    def evaluate(self) -> TrendEvaluation:
        result = TrendEvaluation()
        trends = self._store.load_trends()
        metadata = self._store.load_metadata()

        result.total_trends = len(trends)
        result.has_trends = result.total_trends > 0

        if trends:
            scores = [t.metrics.trend_score for t in trends]
            result.avg_trend_score = round(sum(scores) / len(scores), 4)

            for t in trends:
                if t.trend_type.value == "growing":
                    result.growing_count += 1
                elif t.trend_type.value == "declining":
                    result.declining_count += 1
                elif t.trend_type.value == "emerging":
                    result.emerging_count += 1
                elif t.trend_type.value == "anomaly":
                    result.anomaly_count += 1

            cross_platform = [
                t for t in trends
                if len(t.affected_platforms) >= 2
            ]
            result.cross_platform_count = len(cross_platform)

        result.health = self._compute_health(result)
        return result

    def _compute_health(self, result: TrendEvaluation) -> StageHealth:
        score = 100.0
        warnings: list[str] = []
        metrics: dict[str, float] = {}

        metrics["total_trends"] = float(result.total_trends)
        metrics["growing_count"] = float(result.growing_count)
        metrics["declining_count"] = float(result.declining_count)
        metrics["emerging_count"] = float(result.emerging_count)
        metrics["avg_score"] = result.avg_trend_score

        if not result.has_trends:
            score -= 50
            warnings.append("No trends detected — run 'trend generate' first")
        elif result.total_trends < 3:
            score -= 20
            warnings.append(f"Very few trends ({result.total_trends})")
        elif result.total_trends < 10:
            score -= 10
            warnings.append(f"Low trend count ({result.total_trends})")

        if result.growing_count == 0 and result.has_trends:
            score -= 10
            warnings.append("No growing trends identified")

        if result.emerging_count == 0 and result.has_trends:
            score -= 5
            warnings.append("No emerging trends identified")

        if result.avg_trend_score < 0.3 and result.has_trends:
            score -= 10
            warnings.append(f"Low average trend score ({result.avg_trend_score:.2f})")

        score = max(0.0, min(100.0, score))
        degradation = 100.0 - score

        return StageHealth(
            score=round(score, 1),
            max_score=100.0,
            weight=1.0,
            metrics=metrics,
            warnings=warnings,
            degradation_pct=round(degradation, 1),
        )
