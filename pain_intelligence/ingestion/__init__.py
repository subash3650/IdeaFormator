"""Data Ingestion Framework for the Pain Intelligence Engine."""

from __future__ import annotations

from pain_intelligence.ingestion.config import IngestionConfig, load_ingestion_config
from pain_intelligence.ingestion.engine import IngestionEngine
from pain_intelligence.ingestion.models import RawDocument, SourceType

__all__ = [
    "IngestionConfig",
    "load_ingestion_config",
    "IngestionEngine",
    "RawDocument",
    "SourceType",
]
