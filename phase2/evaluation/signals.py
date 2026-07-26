from __future__ import annotations

from pathlib import Path

import polars as pl

from phase2.evaluation.config import EvalConfig
from phase2.evaluation.metrics import (
    column_exists,
    compute_distribution,
    safe_divide,
    value_counts,
)
from phase2.evaluation.schema import SignalEvaluation, StageHealth


class SignalEvaluator:
    def __init__(self, knowledge_dir: str | Path = "pain_intelligence/knowledge") -> None:
        self._assets_dir = Path(knowledge_dir) / "assets"
        self._reports_dir = Path("reports")

    def evaluate(self, sig_df: pl.DataFrame | None = None) -> SignalEvaluation:
        if sig_df is None:
            path = self._assets_dir / "problem_signals.parquet"
            if not path.exists():
                return SignalEvaluation()
            sig_df = pl.read_parquet(str(path))

        result = SignalEvaluation()
        total = sig_df.height
        result.accepted_signals = total

        if total == 0:
            result.zero_signal_explanation = self._explain_zero_signals()
            result.health = StageHealth(score=0.0, weight=EvalConfig.weight("signals"), warnings=[result.zero_signal_explanation])
            return result

        # Support distribution
        if column_exists(sig_df, "document_count"):
            vals = sig_df["document_count"].to_list()
            result.support_distribution = compute_distribution([float(v) for v in vals])

        # Confidence distribution
        if column_exists(sig_df, "confidence"):
            cv = sig_df["confidence"].drop_nulls().to_list()
            if cv:
                result.confidence_distribution = compute_distribution([float(v) for v in cv])

        # Category coverage
        if column_exists(sig_df, "category"):
            result.category_coverage = value_counts(sig_df["category"])
            coverage = safe_divide(len(result.category_coverage), max(total, 1))
            result.category_coverage_pct = round(coverage * 100, 2)

        # Try to read diagnostics for filtering info
        self._load_filtering_stats(result)

        result.health = self._compute_health(result)
        return result

    def _load_filtering_stats(self, result: SignalEvaluation) -> None:
        diag_path = self._reports_dir / "problem_signal_diagnostics.json"
        if diag_path.exists():
            import json
            try:
                with open(diag_path) as f:
                    diag = json.load(f)
                result.total_candidates = diag.get("total_candidates", 0) or (
                    result.accepted_signals + sum(diag.get("filtering", {}).values())
                    if "filtering" in diag else result.accepted_signals
                )
                filtering = diag.get("filtering", {})
                if filtering:
                    result.filter_reasons = {
                        k: v for k, v in filtering.items() if isinstance(v, int)
                    }
                    result.filtered_signals = sum(result.filter_reasons.values())
            except Exception:
                pass

    def _explain_zero_signals(self) -> str:
        diag_path = self._reports_dir / "problem_signal_diagnostics.json"
        if diag_path.exists():
            import json
            try:
                with open(diag_path) as f:
                    diag = json.load(f)
                reasons = []
                thresholds = diag.get("adaptive_thresholds", {})
                if thresholds:
                    reasons.append(f"adaptive thresholds: {thresholds}")
                filtering = diag.get("filtering", {})
                if filtering:
                    removed = {k: v for k, v in filtering.items() if isinstance(v, int) and v > 0}
                    if removed:
                        reasons.append(f"filtered: {removed}")
                doc_count = diag.get("document_count", 0)
                reasons.append(f"processed {doc_count} documents")
                return "; ".join(reasons) if reasons else "No signals discovered"
            except Exception:
                pass
        return "No problem signals found — dataset may lack sufficient negative signals"

    def _compute_health(self, ev: SignalEvaluation) -> StageHealth:
        score = 100.0
        warnings: list[str] = []

        if ev.accepted_signals == 0:
            return StageHealth(score=0.0, weight=EvalConfig.weight("signals"), warnings=["No signals discovered"])

        if ev.filtered_signals > ev.accepted_signals * 2:
            score -= 15
            warnings.append(f"High filtering rate: {ev.filtered_signals} filtered vs {ev.accepted_signals} accepted")

        if ev.category_coverage and len(ev.category_coverage) < 3:
            score -= 10
            warnings.append(f"Limited category coverage: {len(ev.category_coverage)} categories")

        if ev.support_distribution.mean < EvalConfig.threshold("min_signal_document_count", 3):
            score -= 10
            warnings.append(f"Low average signal support: {ev.support_distribution.mean:.1f} documents")

        score = max(0.0, min(100.0, score))
        return StageHealth(
            score=round(score, 1),
            weight=EvalConfig.weight("signals"),
            warnings=warnings,
        )
