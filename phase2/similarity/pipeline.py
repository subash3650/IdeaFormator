"""SimilarityPipeline – job coordinator for relationship generation."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase2.similarity.config import SimilarityEngineConfig, load_similarity_config
from phase2.similarity.engine import SimilarityEngine


class SimilarityPipeline:
    """Coordinates relationship generation across all source types.

    Thin wrapper around SimilarityEngine for batch orchestration.
    """

    def __init__(self, config_path: str | Path = "configs/default.yaml") -> None:
        self._config = load_similarity_config(config_path)
        self._engine = SimilarityEngine(self._config)

    @property
    def engine(self) -> SimilarityEngine:
        return self._engine

    @property
    def config(self) -> SimilarityEngineConfig:
        return self._config

    def run(self, force: bool = False) -> dict[str, Any]:
        """Execute the full similarity pipeline."""
        started_at = datetime.now(timezone.utc)
        result = self._engine.generate(force=force)
        result["started_at"] = started_at.isoformat()
        return result
