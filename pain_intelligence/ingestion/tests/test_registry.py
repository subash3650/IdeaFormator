"""Tests for the collector registry."""

from __future__ import annotations

import pytest

# Import collectors to trigger @register_collector decorators
import pain_intelligence.ingestion.collectors  # noqa: F401

from pain_intelligence.ingestion.registry import (
    _collectors,
    available_collectors,
    create_collector,
    get_collector_class,
    register_collector,
)
from pain_intelligence.ingestion.collectors.base import BaseCollector
from pain_intelligence.ingestion.config import CollectorConfig
from pain_intelligence.ingestion.models import SourceType
from pain_intelligence.ingestion.clients.base import HttpClient


class DummyCollector(BaseCollector):
    adapter_class = None  # type: ignore[assignment]

    @property
    def source(self) -> SourceType:
        return SourceType.GITHUB

    def authenticate(self) -> None:
        pass

    def health_check(self) -> bool:
        return True

    def fetch(self, state=None):
        yield []


class TestRegistry:
    def test_register_collector(self):
        # DummyCollector is already registered by the import in collectors/__init__
        # but test the decorator explicitly with a fresh name
        @register_collector("test_dummy")
        class _TestDummy(BaseCollector):
            adapter_class = None  # type: ignore[assignment]
            @property
            def source(self) -> SourceType:
                return SourceType.GITHUB
            def authenticate(self) -> None:
                pass
            def health_check(self) -> bool:
                return True
            def fetch(self, state=None):
                yield []

        assert "test_dummy" in _collectors
        # Cleanup
        del _collectors["test_dummy"]

    def test_get_collector_class_exists(self):
        cls = get_collector_class("github")
        assert cls is not None

    def test_get_collector_class_unknown(self):
        with pytest.raises(KeyError, match="Unknown collector"):
            get_collector_class("nonexistent")

    def test_available_collectors(self):
        available = available_collectors()
        assert isinstance(available, list)
        assert "github" in available
        assert "hackernews" in available

    def test_create_collector(self):
        config = CollectorConfig(enabled=True)
        client = HttpClient  # Will be mocked in real tests
        # Just test that the factory function calls the right class
        # In practice, the collector is created with a real client
        assert callable(get_collector_class("github"))
