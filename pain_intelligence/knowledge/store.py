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

from pain_intelligence import PIPELINE_VERSION, SCHEMA_VERSION
from pain_intelligence.knowledge.exceptions import MissingAssetError, StaleAssetError
from pain_intelligence.knowledge.metadata import (
    get_run_id_from_asset,
    make_asset_metadata,
    write_parquet_with_metadata,
)
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

SEEDS = {"entities", "patterns", "taxonomy", "problem_signals", "generic_sentiment"}
ASSETS = {"observations", "evidence", "problem_signals"}

# Schema definitions for empty parquet writing
EMPTY_SCHEMAS: dict[str, dict[str, type]] = {
    "observations": {
        "observation_id": str,
        "type": str,
        "value": str,
        "document_id": str,
        "platform": str,
        "rating": float,
        "country": str,
        "text_snippet": str,
        "extractor": str,
        "method": str,
        "confidence": float,
        "entity": str,
        "entity_type": str,
        "category": str,
        "pattern_label": str,
        "canonical_value": str,
        "canonical_source": str,
        "pipeline_version": str,
        "generated_at": str,
    },
    "evidence": {
        "evidence_id": str,
        "signal_key": str,
        "category": str,
        "entity": str,
        "entity_type": str,
        "signal_text": str,
        "observation_count": int,
        "document_count": int,
        "avg_rating": float,
        "platform_distribution": str,
        "country_distribution": str,
        "observation_ids": str,
        "top_snippets": str,
        "confidence": float,
        "aggregation_strategy": str,
        "pipeline_version": str,
        "generated_at": str,
    },
    "problem_signals": {
        "signal_key": str,
        "category": str,
        "entity": str,
        "entity_type": str,
        "country": str,
        "signal_text": str,
        "document_count": int,
        "avg_rating": float,
        "evidence_ids": str,
        "observation_count": int,
        "confidence": float,
        "pipeline_version": str,
        "generated_at": str,
    },
}


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
        self._run_id: str = ""
        self._knowledge_version: str = "0.1.0"

    # ── Run ID ─────────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._run_id

    @run_id.setter
    def run_id(self, value: str) -> None:
        self._run_id = value

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
        data = self.load_seed("entities")
        return data.get("entities", [])

    def load_patterns(self) -> list[dict[str, Any]]:
        data = self.load_seed("patterns")
        result: list[dict[str, Any]] = []
        for category, pattern_list in data.get("patterns", {}).items():
            for p in pattern_list:
                p["category"] = category
                result.append(p)
        return result

    def load_taxonomy(self) -> dict[str, dict[str, Any]]:
        data = self.load_seed("taxonomy")
        return data.get("categories", {})

    def load_problem_signals(self) -> dict[str, dict[str, Any]]:
        data = self.load_seed("problem_signals")
        return data.get("signals", {})

    def load_generic_sentiment(self) -> list[str]:
        data = self.load_seed("generic_sentiment")
        return data.get("phrases", [])

    # ── Assets ─────────────────────────────────────────────────────

    def write_asset(
        self,
        name: str,
        df: pl.DataFrame,
        input_checksum: str = "",
        input_document_count: int = 0,
    ) -> Path:
        """Write a knowledge asset to Parquet, ALWAYS overwriting.

        If the DataFrame is empty, writes an empty Parquet file with
        the correct schema and embedded metadata.
        """
        if name not in ASSETS:
            raise ValueError(f"Unknown asset: {name}. Available: {ASSETS}")

        # Validate no empty struct columns
        for col_name, dtype in df.schema.items():
            if isinstance(dtype, pl.Struct) and not dtype.fields:
                raise ValueError(
                    f"Column '{col_name}' has type Struct({{}}) — "
                    "empty struct cannot be serialized to Parquet."
                )

        path = self.assets_dir / f"{name}.parquet"

        # If empty, create an empty DataFrame with the correct schema
        if df.height == 0:
            schema = EMPTY_SCHEMAS.get(name, {})
            df = pl.DataFrame({col: pl.Series(col, [], dtype=typ) for col, typ in schema.items()})

        metadata = make_asset_metadata(
            run_id=self._run_id,
            input_checksum=input_checksum,
            input_document_count=input_document_count,
            record_count=df.height,
        )
        result_path = write_parquet_with_metadata(df, path, metadata=metadata)
        logger.info("Wrote {} -> {} ({} rows, run_id={})", name, result_path, df.height, self._run_id)
        return result_path

    def read_asset(
        self,
        name: str,
        validate_run_id: bool = False,
    ) -> pl.DataFrame:
        """Read a knowledge asset from Parquet.

        If validate_run_id is True and the store has a run_id set,
        raises StaleAssetError if the asset's run_id does not match.
        """
        if name not in ASSETS:
            raise ValueError(f"Unknown asset: {name}. Available: {ASSETS}")
        path = self.assets_dir / f"{name}.parquet"
        if not path.exists():
            logger.warning("Asset not found: {}", path)
            raise MissingAssetError(path)

        if validate_run_id and self._run_id:
            asset_run_id = get_run_id_from_asset(path)
            if asset_run_id and asset_run_id != self._run_id:
                raise StaleAssetError(path, self._run_id, asset_run_id)

        return pl.read_parquet(path)

    def read_asset_metadata(self, name: str) -> dict[str, str]:
        """Read embedded metadata from an asset without loading the data."""
        from pain_intelligence.knowledge.metadata import read_parquet_metadata

        if name not in ASSETS:
            raise ValueError(f"Unknown asset: {name}. Available: {ASSETS}")
        path = self.assets_dir / f"{name}.parquet"
        if not path.exists():
            return {}
        return read_parquet_metadata(path)

    def asset_exists(self, name: str) -> bool:
        path = self.assets_dir / f"{name}.parquet"
        return path.exists()

    # ── Versioning ─────────────────────────────────────────────────

    @property
    def pipeline_version(self) -> str:
        return PIPELINE_VERSION

    def get_manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "knowledge_version": self._knowledge_version,
            "run_id": self._run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "assets": {},
        }
        for name in ASSETS:
            path = self.assets_dir / f"{name}.parquet"
            if path.exists():
                meta = {}
                try:
                    from pain_intelligence.knowledge.metadata import read_parquet_metadata
                    meta = read_parquet_metadata(path)
                except Exception:
                    pass
                manifest["assets"][name] = {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    **meta,
                }
        return manifest

    @staticmethod
    def compute_checksum(path: str | Path) -> str:
        path = Path(path)
        if not path.exists():
            return ""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
