"""ClusteringPipeline — high-level orchestration entry point."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from phase2.clustering.config import ClusteringConfig, load_clustering_config
from phase2.clustering.engine import ClusteringEngine


class ClusteringPipeline:
    """High-level pipeline orchestrator for the clustering module.

    Wraps ClusteringEngine for simple invocation with config file path.
    """

    def __init__(self, config_path: str | Path = "configs/default.yaml") -> None:
        self._config_path = Path(config_path)
        self._config = load_clustering_config(self._config_path)
        self._engine = ClusteringEngine(self._config)

    def run(self, force: bool = False) -> dict[str, Any]:
        """Run the full clustering pipeline."""
        return self._engine.generate(force=force)

    def stats(self) -> dict[str, Any]:
        """Return cluster statistics."""
        return self._engine.stats()

    def search(self, query_id: str) -> list[dict[str, Any]]:
        """Search clusters by ID."""
        results = self._engine.search_clusters(query_id)
        return [r.model_dump(mode="json") for r in results]

    def verify(self) -> dict[str, Any]:
        """Verify cluster integrity."""
        return self._engine.verify()
