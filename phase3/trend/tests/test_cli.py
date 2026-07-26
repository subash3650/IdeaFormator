"""Tests for Trend CLI commands."""

from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner

import polars as pl

from phase3.trend.cli import trend_app
from phase3.trend.snapshot import TrendSnapshotBuilder

runner = CliRunner()


class TestTrendCLI:
    def test_stats_empty(self, tmp_path: Path) -> None:
        result = runner.invoke(trend_app, [
            "stats",
            "--knowledge-dir", str(tmp_path),
        ])
        assert result.exit_code == 0

    def test_top_empty(self, tmp_path: Path) -> None:
        result = runner.invoke(trend_app, [
            "top",
            "--knowledge-dir", str(tmp_path),
        ])
        assert result.exit_code == 0

    def test_export_empty(self, tmp_path: Path) -> None:
        result = runner.invoke(trend_app, [
            "export",
            "--knowledge-dir", str(tmp_path),
        ])
        assert result.exit_code == 0

    def test_generate_no_snapshots(self, tmp_path: Path) -> None:
        result = runner.invoke(trend_app, [
            "generate",
            "--knowledge-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "0" in result.output

    def test_snapshot_create(self, tmp_path: Path) -> None:
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"id": ["1"]}).write_parquet(str(assets_dir / "observations.parquet"))

        result = runner.invoke(trend_app, [
            "snapshot",
            "--knowledge-dir", str(tmp_path),
            "--run-id", "test_run",
        ])
        assert result.exit_code == 0
        assert "Snapshot ID" in result.output

    def test_show_nonexistent(self, tmp_path: Path) -> None:
        result = runner.invoke(trend_app, [
            "show", "nonexistent_id",
            "--knowledge-dir", str(tmp_path),
        ])
        assert result.exit_code == 1
        assert "not found" in result.output
