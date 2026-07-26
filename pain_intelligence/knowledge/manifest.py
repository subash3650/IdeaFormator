"""Pipeline manifest — source of truth for all generated assets.

Every pipeline run produces a single manifest file (pipeline_manifest.json)
that records every generated asset, its run_id, checksum, record count,
generation timestamp, and dependency relationships.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pain_intelligence.knowledge.exceptions import StaleAssetError


PIPELINE_VERSION = "1.5.0"
SCHEMA_VERSION = "1.0.0"
MANIFEST_FILENAME = "pipeline_manifest.json"


def generate_run_id() -> str:
    """Generate a deterministic run_id from the current UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def compute_checksum(path: str | Path) -> str:
    """Compute SHA-256 hex digest (first 16 chars) of a file."""
    import hashlib
    path = Path(path)
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class PipelineManifest:
    """Tracks all generated assets across a pipeline run.

    The manifest is the source of truth for:
      - Which run_id this pipeline execution belongs to
      - What assets were generated
      - Asset sizes, record counts, checksums
      - Dependencies between stages
    """

    def __init__(self, base_dir: str | Path = "pain_intelligence/knowledge") -> None:
        self._base_dir = Path(base_dir)
        self._path = self._base_dir / MANIFEST_FILENAME
        self._data: dict[str, Any] = self._load()

    # ── Public API ───────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._data.get("run_id", "")

    @run_id.setter
    def run_id(self, value: str) -> None:
        self._data["run_id"] = value

    @property
    def path(self) -> Path:
        return self._path

    def start_run(self, dataset_path: str | Path | None = None) -> str:
        """Begin a new pipeline run. Generates a fresh run_id.

        If dataset_path is provided, records its checksum and document count.
        """
        run_id = generate_run_id()
        self._data = {
            "run_id": run_id,
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "dataset_checksum": "",
            "dataset_document_count": 0,
            "dataset_path": str(dataset_path) if dataset_path else "",
            "assets": {},
            "dependency_graph": self._default_dependency_graph(),
            "stages": {},
        }
        if dataset_path:
            ds_path = Path(dataset_path)
            if ds_path.exists():
                self._data["dataset_checksum"] = compute_checksum(ds_path)
                import polars as pl
                try:
                    df = pl.read_parquet(str(ds_path))
                    self._data["dataset_document_count"] = df.height
                except Exception:
                    pass
        self._data["run_id"] = run_id
        return run_id

    def register_asset(
        self,
        name: str,
        path: str | Path,
        record_count: int = 0,
        stage: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a generated asset in the manifest."""
        p = Path(path)
        entry: dict[str, Any] = {
            "path": str(p),
            "run_id": self.run_id,
            "record_count": record_count,
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "checksum": compute_checksum(p) if p.exists() else "",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
        }
        if metadata:
            entry["metadata"] = metadata
        self._data["assets"][name] = entry

        # Mark stage completed
        if stage:
            self._data.setdefault("stages", {})[stage] = {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

    def get_asset(self, name: str) -> dict[str, Any] | None:
        """Get manifest entry for a named asset."""
        return self._data.get("assets", {}).get(name)

    def get_stage_status(self, stage: str) -> str:
        """Return the status of a pipeline stage."""
        s = self._data.get("stages", {}).get(stage, {})
        return s.get("status", "not_started")

    def validate_upstream(
        self,
        asset_name: str,
        asset_path: str | Path,
    ) -> None:
        """Validate that an upstream asset belongs to the current run.

        Checks the PERSISTED manifest on disk to find the asset's
        previous run_id. Raises StaleAssetError if the asset's run_id
        does not match the current manifest's run_id, or
        MissingAssetError if the file does not exist.
        """
        p = Path(asset_path)
        if not p.exists():
            from pain_intelligence.knowledge.exceptions import MissingAssetError
            raise MissingAssetError(p)

        # First check persisted manifest on disk
        persisted = self._load()
        asset_entry = persisted.get("assets", {}).get(asset_name)
        if asset_entry is None:
            return

        expected_run_id = asset_entry.get("run_id", "")
        if expected_run_id and expected_run_id != self.run_id:
            raise StaleAssetError(
                asset_path=p,
                expected_run_id=self.run_id,
                actual_run_id=expected_run_id,
            )

    def complete_run(self) -> None:
        """Mark the pipeline run as completed."""
        self._data["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._data["elapsed_seconds"] = (
            time.mktime(datetime.now(timezone.utc).timetuple())
            - time.mktime(
                datetime.fromisoformat(self._data.get("started_at", datetime.now(timezone.utc).isoformat())).timetuple()
            )
        ) if self._data.get("started_at") else 0

    def save(self) -> None:
        """Persist the manifest to disk."""
        self._data["schema_version"] = SCHEMA_VERSION
        self._data["pipeline_version"] = PIPELINE_VERSION
        self._base_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    # ── Private ──────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _default_dependency_graph() -> list[dict[str, Any]]:
        return [
            {"from": "processed.parquet", "to": "observations.parquet"},
            {"from": "observations.parquet", "to": "evidence.parquet"},
            {"from": "evidence.parquet", "to": "problem_signals.parquet"},
            {"from": "problem_signals.parquet", "to": "embeddings"},
            {"from": "embeddings", "to": "relationships"},
            {"from": "relationships", "to": "clusters"},
        ]
