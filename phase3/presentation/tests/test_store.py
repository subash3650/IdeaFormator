from __future__ import annotations

from pathlib import Path

import pytest

from phase3.presentation.schema import (
    PresentationModel,
    ReportFormat,
    ReportIndex,
    ReportIndexEntry,
    ReportOutput,
    ReportType,
)
from phase3.presentation.store import PresentationStore


@pytest.fixture
def store(tmp_path: Path) -> PresentationStore:
    return PresentationStore(tmp_path)


def _make_output(
    report_id: str = "r1",
    report_type: ReportType = ReportType.executive_summary,
    title: str = "Test",
    formats: list[ReportFormat] | None = None,
) -> ReportOutput:
    entry = ReportIndexEntry(
        report_id=report_id,
        report_type=report_type,
        title=title,
        generated_at="2024-01-01T00:00:00",
    )
    return ReportOutput(
        report_id=report_id,
        report_type=report_type,
        title=title,
        generated_at="2024-01-01T00:00:00",
        sections_count=3,
        charts_count=1,
        formats=formats or [ReportFormat.json],
        index_entry=entry,
        elapsed_seconds=1.5,
    )


def _make_model(report_id: str = "r1") -> PresentationModel:
    return PresentationModel(
        report_id=report_id,
        report_type=ReportType.executive_summary,
        title="Test",
        generated_at="2024-01-01T00:00:00",
    )


class TestPresentationStorePaths:
    def test_reports_dir_created(self, tmp_path: Path) -> None:
        s = PresentationStore(tmp_path)
        assert (tmp_path / "reports").exists()

    def test_reports_path(self, tmp_path: Path) -> None:
        s = PresentationStore(tmp_path)
        assert s.reports_path == tmp_path / "reports" / "reports.parquet"

    def test_metadata_path(self, tmp_path: Path) -> None:
        s = PresentationStore(tmp_path)
        assert s.metadata_path == tmp_path / "reports" / "report_metadata.json"

    def test_manifest_path(self, tmp_path: Path) -> None:
        s = PresentationStore(tmp_path)
        assert s.manifest_path == tmp_path / "reports" / "report_manifest.json"

    def test_index_path(self, tmp_path: Path) -> None:
        s = PresentationStore(tmp_path)
        assert s.index_path == tmp_path / "reports" / "report_index.json"

    def test_content_path(self, tmp_path: Path) -> None:
        s = PresentationStore(tmp_path)
        assert s._content_path("abc123") == tmp_path / "reports" / "abc123" / "report_content.json"


class TestSaveAndLoadReport:
    def test_save_report(self, store: PresentationStore) -> None:
        output = _make_output()
        path = store.save_report(output)
        assert path.exists()
        assert path.name == "reports.parquet"

    def test_load_empty(self, store: PresentationStore) -> None:
        assert store.load_all() == []

    def test_save_and_load_one(self, store: PresentationStore) -> None:
        output = _make_output()
        store.save_report(output)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].report_id == "r1"
        assert loaded[0].title == "Test"

    def test_save_and_load_multiple(self, store: PresentationStore) -> None:
        store.save_report(_make_output("r1"))
        store.save_report(_make_output("r2", ReportType.investor, "Investor"))
        loaded = store.load_all()
        assert len(loaded) == 2
        ids = {r.report_id for r in loaded}
        assert ids == {"r1", "r2"}

    def test_save_replaces_existing(self, store: PresentationStore) -> None:
        store.save_report(_make_output("r1", title="Original"))
        store.save_report(_make_output("r1", title="Updated"))
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].title == "Updated"

    def test_round_trip_all_fields(self, store: PresentationStore) -> None:
        entry = ReportIndexEntry(
            report_id="r1",
            report_type=ReportType.technology_landscape,
            title="Tech Report",
            generated_at="2024-06-01T12:00:00",
            tags=["ai", "ml"],
            companies=["Acme"],
            formats=[ReportFormat.json, ReportFormat.html],
            sections=[s for s in []],
        )
        output = ReportOutput(
            report_id="r1",
            report_type=ReportType.technology_landscape,
            title="Tech Report",
            generated_at="2024-06-01T12:00:00",
            sections_count=5,
            charts_count=3,
            formats=[ReportFormat.json, ReportFormat.html, ReportFormat.markdown],
            checksums={"json": "abc123"},
            index_entry=entry,
            elapsed_seconds=2.5,
        )
        store.save_report(output)
        loaded = store.load_all()
        assert len(loaded) == 1
        l = loaded[0]
        assert l.report_id == "r1"
        assert l.report_type == ReportType.technology_landscape
        assert l.sections_count == 5
        assert l.charts_count == 3
        assert ReportFormat.json in l.formats
        assert l.checksums["json"] == "abc123"
        assert l.index_entry.tags == ["ai", "ml"]
        assert l.index_entry.companies == ["Acme"]
        assert l.elapsed_seconds == 2.5

    def test_load_empty_parquet(self, store: PresentationStore) -> None:
        store._write_reports([])
        loaded = store.load_all()
        assert loaded == []


class TestSaveAndLoadContent:
    def test_save_content(self, store: PresentationStore) -> None:
        model = _make_model()
        path = store.save_content(model)
        assert path.exists()
        assert path.name == "report_content.json"

    def test_load_content(self, store: PresentationStore) -> None:
        model = _make_model("r1")
        store.save_content(model)
        loaded = store.load_content("r1")
        assert loaded is not None
        assert loaded.report_id == "r1"
        assert loaded.title == "Test"

    def test_load_content_missing(self, store: PresentationStore) -> None:
        loaded = store.load_content("nonexistent")
        assert loaded is None

    def test_content_round_trip_with_sections(self, store: PresentationStore) -> None:
        from phase3.presentation.schema import ReportSection, SectionType

        section = ReportSection(section_type=SectionType.top_findings, title="Findings", order=0)
        model = _make_model("r2")
        model = PresentationModel(
            report_id="r2",
            report_type=model.report_type,
            title=model.title,
            generated_at=model.generated_at,
            sections=[section],
            tags=["ai"],
        )
        store.save_content(model)
        loaded = store.load_content("r2")
        assert loaded is not None
        assert len(loaded.sections) == 1
        assert loaded.sections[0].title == "Findings"
        assert "ai" in loaded.tags

    def test_content_round_trip_with_assets(self, store: PresentationStore) -> None:
        from phase3.presentation.schema import ReportAssets

        assets = ReportAssets(metrics={"score": 85.0})
        model = _make_model("r3")
        model = PresentationModel(
            report_id="r3",
            report_type=model.report_type,
            title=model.title,
            generated_at=model.generated_at,
            assets=assets,
        )
        store.save_content(model)
        loaded = store.load_content("r3")
        assert loaded is not None
        assert loaded.assets.metrics["score"] == 85.0


class TestMetadataAndManifest:
    def test_save_metadata(self, store: PresentationStore) -> None:
        path = store.save_metadata({"total_reports": 5, "version": "1.0"})
        assert path.exists()
        loaded = store.load_metadata()
        assert loaded is not None
        assert loaded["total_reports"] == 5

    def test_load_metadata_missing(self, store: PresentationStore) -> None:
        assert store.load_metadata() is None

    def test_save_manifest(self, store: PresentationStore) -> None:
        path = store.save_manifest({"run_id": "run123", "generated_at": "2024-01-01"})
        assert path.exists()
        loaded = store.load_manifest()
        assert loaded is not None
        assert loaded["run_id"] == "run123"

    def test_load_manifest_missing(self, store: PresentationStore) -> None:
        assert store.load_manifest() is None


class TestIndex:
    def test_save_and_load_index(self, store: PresentationStore) -> None:
        entry = ReportIndexEntry(
            report_id="r1",
            report_type=ReportType.executive_summary,
            title="Test",
            generated_at="2024-01-01T00:00:00",
            tags=["ai"],
        )
        index = ReportIndex(entries={"r1": entry}, by_tag={"ai": ["r1"]})
        store.save_index(index)
        loaded = store.load_index()
        assert loaded is not None
        assert "r1" in loaded.entries
        assert loaded.by_tag["ai"] == ["r1"]

    def test_load_index_missing(self, store: PresentationStore) -> None:
        assert store.load_index() is None


class TestChecksums:
    def test_checksums_empty(self, store: PresentationStore) -> None:
        cs = store.checksums()
        assert cs == {}

    def test_checksums_after_save(self, store: PresentationStore) -> None:
        store.save_report(_make_output())
        store.save_metadata({"total": 1})
        store.save_manifest({"run": "1"})
        cs = store.checksums()
        assert "reports.parquet" in cs
        assert "report_metadata.json" in cs
        assert "report_manifest.json" in cs

    def test_checksums_length(self, store: PresentationStore) -> None:
        store.save_report(_make_output())
        cs = store.checksums()
        assert len(cs["reports.parquet"]) == 16


class TestCountAndExists:
    def test_count_empty(self, store: PresentationStore) -> None:
        assert store.count() == 0

    def test_count_after_save(self, store: PresentationStore) -> None:
        store.save_report(_make_output("r1"))
        assert store.count() == 1

    def test_exists(self, store: PresentationStore) -> None:
        model = _make_model("r1")
        store.save_content(model)
        assert store.exists("r1") is True

    def test_not_exists(self, store: PresentationStore) -> None:
        assert store.exists("nonexistent") is False
