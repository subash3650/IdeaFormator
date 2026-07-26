"""Tests for the IngestionEngine orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pain_intelligence.ingestion.config import CollectorConfig, IngestionConfig
from pain_intelligence.ingestion.engine import IngestionEngine
from pain_intelligence.ingestion.models import SourceType


class TestIngestionEngine:
    def test_engine_initialization(self, tmp_dir: Path):
        config = IngestionConfig(
            output_base=tmp_dir,
            collectors={
                "github": CollectorConfig(enabled=True),
            },
        )
        engine = IngestionEngine(config)
        assert engine._config.output_base == tmp_dir

    def test_engine_verify(self, tmp_dir: Path):
        config = IngestionConfig(
            output_base=tmp_dir,
            collectors={
                "github": CollectorConfig(enabled=True),
            },
        )
        engine = IngestionEngine(config)
        result = engine.verify()
        assert "github" in result
        assert result["github"]["state_loaded"] is True
        assert result["github"]["total_collected"] == 0

    def test_engine_stats(self, tmp_dir: Path):
        config = IngestionConfig(
            output_base=tmp_dir,
            collectors={
                "github": CollectorConfig(enabled=True),
                "hackernews": CollectorConfig(enabled=True),
            },
        )
        engine = IngestionEngine(config)
        stats = engine.stats()
        assert "github" in stats
        assert "hackernews" in stats
