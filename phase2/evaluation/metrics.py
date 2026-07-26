from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import polars as pl

from phase2.evaluation.schema import DistributionStats, HistogramBin


def safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_distribution(values: Sequence[float], bins: int = 10) -> DistributionStats:
    arr = list(values)
    if not arr:
        return DistributionStats()

    sorted_vals = sorted(arr)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    variance = sum((x - mean) ** 2 for x in sorted_vals) / n
    std = math.sqrt(variance)

    def percentile(p: float) -> float:
        idx = int(p * (n - 1))
        return sorted_vals[idx]

    hist = _auto_histogram(sorted_vals, bins)

    return DistributionStats(
        min=float(sorted_vals[0]),
        max=float(sorted_vals[-1]),
        mean=round(mean, 4),
        median=float(sorted_vals[n // 2]),
        std=round(std, 4),
        p25=percentile(0.25),
        p75=percentile(0.75),
        histogram=hist,
    )


def _auto_histogram(sorted_vals: list[float], max_bins: int = 10) -> list[HistogramBin]:
    if not sorted_vals:
        return []
    mn, mx = sorted_vals[0], sorted_vals[-1]
    if mn == mx:
        return [HistogramBin(label=f"{mn:.4f}", count=len(sorted_vals), percentage=100.0)]

    n = len(sorted_vals)
    n_bins = min(max_bins, n)
    bin_width = (mx - mn) / n_bins
    bins: list[HistogramBin] = []
    for i in range(n_bins):
        lo = mn + i * bin_width
        hi = lo + bin_width if i < n_bins - 1 else mx + 0.0001
        count = sum(1 for v in sorted_vals if lo <= v < hi)
        label = f"{lo:.4f}-{hi:.4f}" if n_bins > 1 else f"{mn:.4f}"
        bins.append(HistogramBin(label=label, count=count, percentage=round(count / n * 100, 2)))
    return bins


def column_exists(df: pl.DataFrame, col: str) -> bool:
    return col in df.columns


def safe_column(df: pl.DataFrame, col: str, default: Any = None) -> pl.Series:
    if column_exists(df, col):
        return df[col]
    return pl.Series(col, [])


def non_null_count(s: pl.Series) -> int:
    return s.drop_nulls().len()


def null_count(s: pl.Series) -> int:
    return s.is_null().sum()


def uniqueness_ratio(s: pl.Series) -> float:
    total = s.len()
    if total == 0:
        return 0.0
    return s.n_unique() / total


def value_counts(s: pl.Series) -> dict[str, int]:
    if s.len() == 0:
        return {}
    counts = s.value_counts()
    val_col = s.name
    cnt_col = [c for c in counts.columns if c != s.name][0]
    return {str(row[val_col]): int(row[cnt_col]) for row in counts.iter_rows(named=True)}


def entropy(s: pl.Series) -> float:
    counts = value_counts(s)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
