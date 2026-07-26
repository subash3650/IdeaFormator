from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from phase3.presentation.config import PresentationConfig
from phase3.presentation.schema import (
    PresentationModel,
    ReportFormat,
    ReportIndex,
    ReportIndexEntry,
    ReportOutput,
    ReportType,
)
from pain_intelligence.knowledge.metadata import write_parquet_with_metadata

PARQUET_SCHEMA: dict[str, pl.DataType] = {
    "report_id": pl.Utf8,
    "report_type": pl.Utf8,
    "title": pl.Utf8,
    "generated_at": pl.Utf8,
    "sections_count": pl.Int64,
    "charts_count": pl.Int64,
    "formats": pl.Utf8,
    "checksums": pl.Utf8,
    "index_entry": pl.Utf8,
    "elapsed_seconds": pl.Float64,
}


def _output_to_row(output: ReportOutput) -> dict[str, Any]:
    return {
        "report_id": output.report_id,
        "report_type": output.report_type.value,
        "title": output.title,
        "generated_at": output.generated_at,
        "sections_count": output.sections_count,
        "charts_count": output.charts_count,
        "formats": json.dumps([f.value for f in output.formats]),
        "checksums": json.dumps(output.checksums),
        "index_entry": json.dumps(output.index_entry.model_dump(mode="json")),
        "elapsed_seconds": output.elapsed_seconds,
    }


def _row_to_output(row: dict[str, Any]) -> ReportOutput:
    return ReportOutput(
        report_id=row.get("report_id", ""),
        report_type=ReportType(row.get("report_type", "executive_summary")),
        title=row.get("title", ""),
        generated_at=row.get("generated_at", ""),
        sections_count=int(row.get("sections_count", 0)),
        charts_count=int(row.get("charts_count", 0)),
        formats=[ReportFormat(f) for f in json.loads(row.get("formats", "[]"))],
        checksums=json.loads(row.get("checksums", "{}")),
        index_entry=ReportIndexEntry(**json.loads(row.get("index_entry", "{}"))),
        elapsed_seconds=float(row.get("elapsed_seconds", 0.0)),
    )


class PresentationStore:
    def __init__(self, base_path: Path | str) -> None:
        self._base = Path(base_path)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _reports_dir(self) -> Path:
        return self._base / "reports"

    @property
    def reports_path(self) -> Path:
        return self._reports_dir / "reports.parquet"

    @property
    def metadata_path(self) -> Path:
        return self._reports_dir / "report_metadata.json"

    @property
    def manifest_path(self) -> Path:
        return self._reports_dir / "report_manifest.json"

    @property
    def index_path(self) -> Path:
        return self._reports_dir / "report_index.json"

    def _content_dir(self, report_id: str) -> Path:
        return self._reports_dir / report_id

    def _content_path(self, report_id: str) -> Path:
        return self._content_dir(report_id) / "report_content.json"

    # ---- Save ----

    def save_report(self, output: ReportOutput) -> Path:
        existing = self.load_all()
        existing = [o for o in existing if o.report_id != output.report_id]
        existing.append(output)
        self._write_reports(existing)
        return self.reports_path

    def save_content(self, model: PresentationModel) -> Path:
        content_dir = self._content_dir(model.report_id)
        content_dir.mkdir(parents=True, exist_ok=True)
        path = self._content_path(model.report_id)
        data = model.model_dump(mode="json")
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_metadata(self, metadata: dict[str, Any]) -> Path:
        with open(str(self.metadata_path), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        return self.metadata_path

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        with open(str(self.manifest_path), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        return self.manifest_path

    def save_index(self, index: ReportIndex) -> Path:
        with open(str(self.index_path), "w", encoding="utf-8") as f:
            json.dump(index.model_dump(mode="json"), f, indent=2, default=str)
        return self.index_path

    # ---- Load ----

    def load_all(self) -> list[ReportOutput]:
        if not self.reports_path.exists():
            return []
        df = pl.read_parquet(str(self.reports_path))
        return [_row_to_output(row) for row in df.iter_rows(named=True)]

    def load_content(self, report_id: str) -> PresentationModel | None:
        path = self._content_path(report_id)
        if not path.exists():
            return None
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return PresentationModel(**data)

    def load_metadata(self) -> dict[str, Any] | None:
        if not self.metadata_path.exists():
            return None
        with open(str(self.metadata_path), "r", encoding="utf-8") as f:
            return json.load(f)

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        with open(str(self.manifest_path), "r", encoding="utf-8") as f:
            return json.load(f)

    def load_index(self) -> ReportIndex | None:
        if not self.index_path.exists():
            return None
        with open(str(self.index_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return ReportIndex(**data)

    # ---- Count / Checksums / Exists ----

    def count(self) -> int:
        return len(self.load_all())

    def checksums(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for f in [self.reports_path, self.metadata_path, self.manifest_path, self.index_path]:
            if f.exists():
                raw = f.read_bytes()
                result[f.name] = hashlib.sha256(raw).hexdigest()[:16]
        return result

    def exists(self, report_id: str) -> bool:
        return self._content_path(report_id).exists()

    # ---- Internal ----

    def _write_reports(self, outputs: list[ReportOutput]) -> None:
        rows = [_output_to_row(o) for o in outputs]
        if not rows:
            empty = pl.DataFrame(schema={k: v for k, v in PARQUET_SCHEMA.items()})
            write_parquet_with_metadata(empty, str(self.reports_path), metadata={"count": "0"})
            return
        df = pl.DataFrame(rows, schema=PARQUET_SCHEMA)
        metadata = {"count": str(len(rows))}
        write_parquet_with_metadata(df, str(self.reports_path), metadata=metadata)
