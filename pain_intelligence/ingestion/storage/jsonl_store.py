"""JSONL storage for immutable raw document archives."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pain_intelligence.ingestion.models import RawDocument
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class JsonlStore:
    """Writes and reads RawDocuments as JSONL files.

    Output path: {output_base}/raw/{source}/{source}_{YYYYMMDD}.jsonl
    """

    def __init__(self, output_base: Path) -> None:
        self._output_base = output_base

    def write(self, source_name: str, documents: list[RawDocument]) -> Path:
        """Write documents to a JSONL file, appending to today's file."""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        source_dir = self._output_base / "raw" / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        path = source_dir / f"{source_name}_{date_str}.jsonl"
        mode = "a" if path.exists() else "w"

        with open(path, mode, encoding="utf-8") as f:
            for doc in documents:
                line = json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, default=str)
                f.write(line + "\n")

        logger.debug("Wrote {} documents to {}", len(documents), path)
        return path

    def read(self, source_name: str, date_str: str | None = None) -> list[dict[str, Any]]:
        """Read documents from a JSONL file."""
        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

        path = self._output_base / "raw" / source_name / f"{source_name}_{date_str}.jsonl"
        if not path.exists():
            return []

        documents: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(json.loads(line))

        return documents
