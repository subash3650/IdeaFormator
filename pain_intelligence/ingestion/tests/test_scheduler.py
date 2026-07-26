"""Tests for the ingestion scheduler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pain_intelligence.ingestion.config import CollectorConfig, IngestionConfig
from pain_intelligence.ingestion.engine import IngestionEngine
from pain_intelligence.ingestion.scheduler.scheduler import IngestionScheduler


class TestIngestionScheduler:
    def test_run_once(self, tmp_dir: Path):
        config = IngestionConfig(
            output_base=tmp_dir,
            schedule="once",
            collectors={
                "github": CollectorConfig(enabled=False),
            },
        )
        engine = IngestionEngine(config)
        scheduler = IngestionScheduler(engine, config)

        result = scheduler.run_once()
        assert result["status"] == "completed"
        assert "elapsed_seconds" in result

    def test_scheduler_initialization(self, tmp_dir: Path):
        config = IngestionConfig(
            output_base=tmp_dir,
            schedule="daily",
            collectors={},
        )
        engine = IngestionEngine(config)
        scheduler = IngestionScheduler(engine, config)
        assert scheduler._config.schedule == "daily"
        assert scheduler._scheduler is None
