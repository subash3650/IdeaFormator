"""Tests for state management."""

from __future__ import annotations

from pathlib import Path

import pytest

from pain_intelligence.ingestion.models import SourceType, SyncState
from pain_intelligence.ingestion.state.manager import StateManager


class TestStateManager:
    def test_load_default_state(self, tmp_dir: Path):
        manager = StateManager(tmp_dir)
        state = manager.load_state(SourceType.GITHUB)
        assert state.source == SourceType.GITHUB
        assert state.last_sync is None
        assert state.failure_count == 0

    def test_save_and_load(self, tmp_dir: Path):
        manager = StateManager(tmp_dir)
        state = SyncState(source=SourceType.GITHUB, total_collected=100)
        manager.save_state(state)

        loaded = manager.load_state(SourceType.GITHUB)
        assert loaded.total_collected == 100

    def test_record_success(self, tmp_dir: Path):
        manager = StateManager(tmp_dir)
        manager.record_success(SourceType.GITHUB, 50)

        state = manager.load_state(SourceType.GITHUB)
        assert state.total_collected == 50
        assert state.last_sync is not None
        assert state.failure_count == 0

    def test_record_failure(self, tmp_dir: Path):
        manager = StateManager(tmp_dir)
        manager.record_failure(SourceType.GITHUB, "API timeout")

        state = manager.load_state(SourceType.GITHUB)
        assert state.failure_count == 1
        assert state.last_error == "API timeout"

    def test_record_multiple_failures(self, tmp_dir: Path):
        manager = StateManager(tmp_dir)
        manager.record_failure(SourceType.GITHUB, "Error 1")
        manager.record_failure(SourceType.GITHUB, "Error 2")

        state = manager.load_state(SourceType.GITHUB)
        assert state.failure_count == 2
        assert state.last_error == "Error 2"

    def test_reset_state(self, tmp_dir: Path):
        manager = StateManager(tmp_dir)
        manager.record_success(SourceType.GITHUB, 100)
        manager.reset_state(SourceType.GITHUB)

        state = manager.load_state(SourceType.GITHUB)
        assert state.total_collected == 0
        assert state.failure_count == 0

    def test_clear_all(self, tmp_dir: Path):
        manager = StateManager(tmp_dir)
        manager.record_success(SourceType.GITHUB, 50)
        manager.record_success(SourceType.HACKERNEWS, 30)
        manager.clear_all()

        assert manager.load_state(SourceType.GITHUB).total_collected == 0
        assert manager.load_state(SourceType.HACKERNEWS).total_collected == 0

    def test_independent_sources(self, tmp_dir: Path):
        manager = StateManager(tmp_dir)
        manager.record_success(SourceType.GITHUB, 100)
        manager.record_failure(SourceType.HACKERNEWS, "Error")

        gh = manager.load_state(SourceType.GITHUB)
        hn = manager.load_state(SourceType.HACKERNEWS)
        assert gh.total_collected == 100
        assert hn.failure_count == 1
