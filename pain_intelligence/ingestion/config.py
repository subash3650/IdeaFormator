"""Configuration loader for the Ingestion Framework."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env
load_dotenv()


class CollectorConfig(BaseModel):
    """Configuration for a specific collector."""

    model_config = {"frozen": True, "extra": "forbid"}

    enabled: bool = True
    api_key_env: str | None = None  # Name of environment variable holding the API Key/Token
    batch_size: int = 100
    max_pages: int = 10
    rate_limit: float = 1.0  # requests per second
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0
    api_version: str = ""

    # Play Store specific fields (ignored by other collectors)
    language: str = "en"
    country: str = "us"
    review_limit: int = 1000
    sort: str = "newest"
    apps: list[str] = Field(default_factory=list)
    apps_config_path: str | None = None


class IngestionConfig(BaseModel):
    """Global configuration for the Ingestion Framework."""

    model_config = {"frozen": True, "extra": "forbid"}

    pipeline_version: str = "0.1.0"
    output_base: Path = Path("pain_intelligence/knowledge")
    schedule: str = "daily"
    log_level: str = "INFO"
    collectors: dict[str, CollectorConfig] = Field(default_factory=dict)

    def resolve_api_key(self, source: str) -> str | None:
        """Resolve the API key for a collector from environment variables."""
        cfg = self.collectors.get(source)
        if cfg and cfg.api_key_env:
            return os.environ.get(cfg.api_key_env)
        return None


def load_ingestion_config(path: str | Path = "configs/ingestion.yaml") -> IngestionConfig:
    """Load configuration from a YAML file."""
    p = Path(path)
    if not p.exists():
        # Fall back to returning default config if file is missing
        return IngestionConfig(
            collectors={
                "github": CollectorConfig(api_key_env="GITHUB_TOKEN", rate_limit=0.5),
                "hackernews": CollectorConfig(api_key_env=None, rate_limit=2.0),
            }
        )

    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Extract the ingestion section if present, otherwise use root
    ingestion_data = data.get("ingestion", data)

    # Convert paths to Path objects if necessary
    if "output_base" in ingestion_data:
        ingestion_data["output_base"] = Path(ingestion_data["output_base"])

    # Parse collectors dict
    collectors_data = ingestion_data.get("collectors", {})
    collectors = {}
    for name, val in collectors_data.items():
        collectors[name] = CollectorConfig(**val)

    ingestion_data["collectors"] = collectors

    return IngestionConfig(**ingestion_data)
