from __future__ import annotations

from pathlib import Path

import polars as pl

from phase2.evaluation.config import EvalConfig
from phase2.evaluation.metrics import (
    column_exists,
    compute_distribution,
    null_count,
    safe_divide,
    value_counts,
)
from phase2.evaluation.schema import DocumentEvaluation, StageHealth


class DocumentEvaluator:
    def __init__(self, knowledge_dir: str | Path = "pain_intelligence/knowledge") -> None:
        self._knowledge_dir = Path(knowledge_dir)

    def _find_dataset(self) -> Path | None:
        candidates = [
            self._knowledge_dir / "processed" / "processed.parquet",
            Path("outputs") / "processed.parquet",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def evaluate(self, df: pl.DataFrame | None = None) -> DocumentEvaluation:
        if df is None:
            path = self._find_dataset()
            if path is None:
                return DocumentEvaluation()
            df = pl.read_parquet(str(path))

        if df.height == 0:
            return DocumentEvaluation()

        total = df.height
        result = DocumentEvaluation(total_documents=total)

        PLATFORM_COLS = ["platform", "source", "source_dataset"]
        platform_col = next((c for c in PLATFORM_COLS if column_exists(df, c)), None)
        if platform_col:
            result.documents_per_source = value_counts(df[platform_col])

        id_col = next((c for c in ["id", "document_id", "external_id"] if column_exists(df, c)), None)
        if id_col:
            result.duplicate_count = total - df[id_col].n_unique()
            result.duplicate_rate = round(safe_divide(result.duplicate_count, total), 4)

        text_col = next((c for c in ["text", "content", "clean_text"] if column_exists(df, c)), None)
        if text_col:
            lengths = [v for v in (df[text_col].str.len_bytes().to_list() if df[text_col].dtype == pl.Utf8 else []) if v is not None]
            if lengths:
                result.avg_document_length = round(sum(lengths) / len(lengths), 2)
                char_lengths = [v for v in (df[text_col].str.len_chars().to_list()) if v is not None]
                if char_lengths:
                    result.avg_document_length_chars = round(sum(char_lengths) / len(char_lengths), 2)
            result.empty_content_count = df[text_col].is_null().sum() + (df[text_col] == "").sum()
            result.empty_content_rate = round(safe_divide(result.empty_content_count, total), 4)

        if column_exists(df, "language"):
            result.language_distribution = value_counts(df["language"])

        _META_FIELDS = ["language", "country", "platform", "rating", "created_at", "author"]
        present_meta = [c for c in _META_FIELDS if column_exists(df, c)]
        if present_meta:
            missing = sum(null_count(df[c]) for c in present_meta)
            result.missing_fields = {c: null_count(df[c]) for c in present_meta}
            total_meta = total * len(present_meta)
            result.metadata_completeness = round(
                safe_divide(total_meta - missing, total_meta), 4
            )

        date_col = next((c for c in ["created_at", "ingested_at"] if column_exists(df, c)), None)
        if date_col:
            dates = df[date_col].drop_nulls()
            if dates.len() > 1:
                try:
                    raw = [str(d)[:10] for d in dates]
                    parsed_dates = []
                    from datetime import datetime as dt_mod
                    for d_str in raw:
                        try:
                            parsed_dates.append(dt_mod.strptime(d_str, "%Y-%m-%d"))
                        except ValueError:
                            pass
                    if len(parsed_dates) > 1:
                        earliest = min(parsed_dates)
                        latest = max(parsed_dates)
                        result.date_range = {
                            "earliest": earliest.strftime("%Y-%m-%d"),
                            "latest": latest.strftime("%Y-%m-%d"),
                        }
                        result.collection_freshness_days = float((latest - earliest).days)
                except Exception:
                    pass

        result.health = self._compute_health(result)
        return result

    def _compute_health(self, ev: DocumentEvaluation) -> StageHealth:
        score = 100.0
        warnings: list[str] = []

        if ev.total_documents == 0:
            return StageHealth(score=0.0, weight=EvalConfig.weight("documents"), warnings=["No documents found"])

        if ev.duplicate_rate > EvalConfig.threshold("duplicate_rate_max", 0.10):
            penalty = min(30, ev.duplicate_rate * 100)
            score -= penalty
            warnings.append(f"High duplicate rate: {ev.duplicate_rate:.1%}")

        if ev.empty_content_rate > 0.05:
            penalty = min(20, ev.empty_content_rate * 100)
            score -= penalty
            warnings.append(f"High empty content rate: {ev.empty_content_rate:.1%}")

        if ev.metadata_completeness < 0.8 and ev.metadata_completeness > 0:
            penalty = min(20, (1 - ev.metadata_completeness) * 50)
            score -= penalty
            warnings.append(f"Low metadata completeness: {ev.metadata_completeness:.1%}")

        if not ev.language_distribution:
            score -= 10
            warnings.append("No language information")

        score = max(0.0, min(100.0, score))
        return StageHealth(
            score=round(score, 1),
            weight=EvalConfig.weight("documents"),
            warnings=warnings,
        )
