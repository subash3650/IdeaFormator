"""Unified store for seed knowledge (YAML) and generated assets (Parquet).

Seeds are versioned, human-readable startup knowledge.
Assets are machine-generated, evolving truth.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

SEEDS = {"entities", "patterns", "taxonomy", "problem_signals", "generic_sentiment"}
ASSETS = {"observations", "evidence", "problem_signals"}


class KnowledgeStore:
    """Unified store for seeds (YAML) and assets (Parquet).
    
    Seeds are read-only, human-maintained bootstrap knowledge.
    Assets are machine-generated, versioned, and evolve over time.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.seeds_dir = self.base_dir / "seeds"
        self.assets_dir = self.base_dir / "assets"
        self.seeds_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline_version: str = "1.5.0"
        self._knowledge_version: str = "0.1.0"

    # ── Seeds ─────────────────────────────────────────────────────

    def load_seed(self, name: str) -> dict[str, Any]:
        """Load a YAML seed file."""
        if name not in SEEDS:
            raise ValueError(f"Unknown seed: {name}. Available: {SEEDS}")
        path = self.seeds_dir / f"{name}.yaml"
        if not path.exists():
            logger.warning("Seed file not found: {}", path)
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_entities(self) -> list[dict[str, Any]]:
        """Load seed entities list."""
        data = self.load_seed("entities")
        return data.get("entities", [])

    def load_patterns(self) -> list[dict[str, Any]]:
        """Load seed patterns."""
        data = self.load_seed("patterns")
        result: list[dict[str, Any]] = []
        for category, pattern_list in data.get("patterns", {}).items():
            for p in pattern_list:
                p["category"] = category
                result.append(p)
        return result

    def load_taxonomy(self) -> dict[str, dict[str, Any]]:
        """Load taxonomy categories."""
        data = self.load_seed("taxonomy")
        return data.get("categories", {})

    def load_problem_signals(self) -> dict[str, dict[str, Any]]:
        """Load canonical problem signal concepts."""
        data = self.load_seed("problem_signals")
        return data.get("signals", {})

    def load_generic_sentiment(self) -> list[str]:
        """Load generic sentiment-only phrases."""
        data = self.load_seed("generic_sentiment")
        return data.get("phrases", [])

    # ── Assets ─────────────────────────────────────────────────────

    def write_asset(self, name: str, df: pl.DataFrame) -> Path:
        """Write a knowledge asset to Parquet."""
        if name not in ASSETS:
            raise ValueError(f"Unknown asset: {name}. Available: {ASSETS}")
        path = self.assets_dir / f"{name}.parquet"
        df.write_parquet(path)
        logger.info("Wrote {} -> {} ({} rows)", name, path, len(df))
        return path

    def read_asset(self, name: str) -> pl.DataFrame:
        """Read a knowledge asset from Parquet."""
        if name not in ASSETS:
            raise ValueError(f"Unknown asset: {name}. Available: {ASSETS}")
        path = self.assets_dir / f"{name}.parquet"
        if not path.exists():
            logger.warning("Asset not found: {}", path)
            return pl.DataFrame()
        return pl.read_parquet(path)

    def asset_exists(self, name: str) -> bool:
        """Check if an asset exists on disk."""
        path = self.assets_dir / f"{name}.parquet"
        return path.exists()

    # ── Versioning ─────────────────────────────────────────────────

    @property
    def pipeline_version(self) -> str:
        return self._pipeline_version

    @pipeline_version.setter
    def pipeline_version(self, v: str) -> None:
        self._pipeline_version = v

    @property
    def knowledge_version(self) -> str:
        return self._knowledge_version

    @knowledge_version.setter
    def knowledge_version(self, v: str) -> None:
        self._knowledge_version = v

    def get_manifest(self) -> dict[str, Any]:
        """Return version manifest for all assets."""
        manifest: dict[str, Any] = {
            "pipeline_version": self.pipeline_version,
            "knowledge_version": self.knowledge_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "assets": {},
        }
        for name in ASSETS:
            path = self.assets_dir / f"{name}.parquet"
            if path.exists():
                df = pl.read_parquet(path, n_rows=0)
                manifest["assets"][name] = {
                    "columns": df.columns,
                    "size_bytes": path.stat().st_size,
                }
        return manifest

    @staticmethod
    def compute_checksum(path: str | Path) -> str:
        """Compute SHA-256 checksum of a file."""
        path = Path(path)
        if not path.exists():
            return ""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]