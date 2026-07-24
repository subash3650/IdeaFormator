"""Export manifest and quality report for an embedding run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from phase2.embeddings.schema import EmbeddingManifest


def write_manifest(
    manifest: EmbeddingManifest,
    output_dir: Path,
) -> Path:
    """Write the embedding manifest as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "embedding_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2, default=str)
    return path


def generate_quality_report(
    df: pl.DataFrame,
    output_dir: Path,
    run_id: str | None = None,
) -> Path:
    """Generate a human-readable quality report."""
    from phase2.embeddings.metrics import compute_stats

    stats = compute_stats(df)

    lines = [
        "=" * 50,
        "Embedding Quality Report",
        "=" * 50,
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Run ID:    {run_id or 'N/A'}",
        "",
        "--- Overview ---",
        f"Total vectors : {stats.total_vectors}",
        f"Dimension     : {stats.dimension}",
        f"Mean norm     : {stats.mean_norm:.6f}",
        f"Std norm      : {stats.std_norm:.6f}",
        "",
        "--- By Source ---",
    ]
    for src, count in sorted(stats.by_source.items()):
        lines.append(f"  {src}: {count}")
    lines.extend([
        "",
        f"Null snippets : {stats.null_text_snippets}",
        "",
        "Checks",
        "------",
        f"  All norms ~1.0 : {'PASS' if abs(stats.mean_norm - 1.0) < 0.01 else 'WARN'}",
        f"  Vectors present: {'PASS' if stats.total_vectors > 0 else 'FAIL'}",
        "=" * 50,
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "embedding_quality_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path