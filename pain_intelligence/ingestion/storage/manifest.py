"""Manifest builder for ingestion run metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pain_intelligence.ingestion.models import IngestionManifest, RawDocument
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

PIPELINE_VERSION = "0.1.0"
SCHEMA_VERSION = "1.0.0"


class ManifestBuilder:
    """Builds and writes ingestion manifests after a collection run.

    Manifest includes:
    - Schema, pipeline, and collector versions
    - Document count
    - Checksum of all document IDs
    - Generated timestamp
    - Output file paths
    """

    def __init__(self, output_base: Path) -> None:
        self._output_base = output_base

    def write(
        self,
        source_name: str,
        documents: list[RawDocument],
        file_paths: list[str] | None = None,
    ) -> Path:
        """Build and write a manifest JSON file."""
        reports_dir = self._output_base / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Compute aggregate checksum from all document IDs
        id_str = "".join(sorted(doc.document_id for doc in documents))
        checksum = hashlib.sha256(id_str.encode("utf-8")).hexdigest()[:16]

        manifest = IngestionManifest(
            pipeline_version=PIPELINE_VERSION,
            schema_version=SCHEMA_VERSION,
            source=documents[0].source if documents else "github",
            document_count=len(documents),
            checksum=checksum,
            file_paths=file_paths or [],
        )

        path = reports_dir / f"{source_name}_manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(mode="json"), f, indent=2, ensure_ascii=False, default=str)

        logger.debug("Wrote manifest to {}", path)
        return path
