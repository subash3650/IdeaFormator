"""Regression tests: Evidence.to_dataframe() must never produce Struct({})."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from pain_intelligence.intelligence.schema import (
    EntityType,
    Evidence,
    ProblemSignal,
)


class TestEvidenceToDataframe:
    """Evidence.to_dataframe() must serialize dict fields to JSON strings
    so Polars never infers Struct({}) which cannot be written to Parquet.
    """

    def test_empty_dict(self) -> None:
        ev = Evidence(
            evidence_id="ev1",
            signal_key="test:value:",
            platform_distribution={},
            country_distribution={},
        )
        df = Evidence.to_dataframe([ev])
        assert df["platform_distribution"].dtype == pl.String
        assert df["country_distribution"].dtype == pl.String
        assert json.loads(df["platform_distribution"][0]) == {}
        assert json.loads(df["country_distribution"][0]) == {}

    def test_populated_dict(self) -> None:
        ev = Evidence(
            evidence_id="ev1",
            signal_key="test:value:",
            platform_distribution={"amazon": 5, "google_play": 3},
            country_distribution={"us": 10, "uk": 2},
        )
        df = Evidence.to_dataframe([ev])
        assert df["platform_distribution"].dtype == pl.String
        assert json.loads(df["platform_distribution"][0]) == {"amazon": 5, "google_play": 3}

    def test_mixed_empty_and_populated(self) -> None:
        ev1 = Evidence(
            evidence_id="ev1",
            signal_key="test:value1:",
            platform_distribution={},
            country_distribution={"us": 1},
        )
        ev2 = Evidence(
            evidence_id="ev2",
            signal_key="test:value2:",
            platform_distribution={"amazon": 3},
            country_distribution={},
        )
        df = Evidence.to_dataframe([ev1, ev2])
        assert df["platform_distribution"].dtype == pl.String
        assert df["country_distribution"].dtype == pl.String
        assert json.loads(df["platform_distribution"][0]) == {}
        assert json.loads(df["platform_distribution"][1]) == {"amazon": 3}
        assert json.loads(df["country_distribution"][0]) == {"us": 1}
        assert json.loads(df["country_distribution"][1]) == {}

    def test_write_parquet_empty_dict(self, tmp_path: Path) -> None:
        ev = Evidence(
            evidence_id="ev1",
            signal_key="test:value:",
            platform_distribution={},
            country_distribution={},
        )
        df = Evidence.to_dataframe([ev])
        path = tmp_path / "evidence.parquet"
        df.write_parquet(path)
        reloaded = pl.read_parquet(path)
        assert reloaded["platform_distribution"].dtype == pl.String
        assert reloaded["country_distribution"].dtype == pl.String

    def test_write_parquet_populated_dict(self, tmp_path: Path) -> None:
        ev = Evidence(
            evidence_id="ev1",
            signal_key="test:value:",
            platform_distribution={"amazon": 10, "google_play": 5},
            country_distribution={"us": 8, "de": 3},
        )
        df = Evidence.to_dataframe([ev])
        path = tmp_path / "evidence.parquet"
        df.write_parquet(path)
        reloaded = pl.read_parquet(path)
        assert reloaded["platform_distribution"].dtype == pl.String

    def test_write_parquet_mixed_dict(self, tmp_path: Path) -> None:
        ev1 = Evidence(
            evidence_id="ev1",
            signal_key="test:value1:",
            platform_distribution={},
            country_distribution={"us": 1},
        )
        ev2 = Evidence(
            evidence_id="ev2",
            signal_key="test:value2:",
            platform_distribution={"google_play": 2},
            country_distribution={},
        )
        df = Evidence.to_dataframe([ev1, ev2])
        path = tmp_path / "evidence.parquet"
        df.write_parquet(path)
        reloaded = pl.read_parquet(path)
        assert len(reloaded) == 2
        assert reloaded["platform_distribution"].dtype == pl.String

    def test_all_fields_no_struct(self) -> None:
        """All columns should be String, Int64, Float64, or List — never Struct."""
        ev = Evidence(
            evidence_id="ev1",
            signal_key="test:value:",
            category="shipping",
            entity="Amazon",
            entity_type=EntityType.COMPANY,
            signal_text="late delivery",
            observation_count=10,
            document_count=8,
            avg_rating=2.5,
            platform_distribution={"amazon": 8},
            country_distribution={"us": 5, "uk": 3},
            observation_ids=["o1", "o2"],
            top_snippets=["bad", "terrible"],
            confidence=0.85,
            aggregation_strategy="rule",
            pipeline_version="1.5.0",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        df = Evidence.to_dataframe([ev])
        for col_name, dtype in df.schema.items():
            assert not isinstance(dtype, pl.Struct), (
                f"Column '{col_name}' has unexpected Struct type: {dtype}"
            )
        # Verify platform_distribution content round-trips
        parsed = json.loads(df["platform_distribution"][0])
        assert parsed == {"amazon": 8}


class TestWriteAssetValidation:
    """KnowledgeStore.write_asset() must reject empty struct columns with clear error."""

    def test_validation_rejects_empty_struct(self) -> None:
        from pain_intelligence.knowledge.store import KnowledgeStore
        store = KnowledgeStore(".")
        df = pl.DataFrame({"x": pl.Series("x", [{}], dtype=pl.Struct)})
        with pytest.raises(ValueError, match="empty struct cannot be serialized"):
            store.write_asset("evidence", df)

    def test_validation_passes_json_serialized(self) -> None:
        from pain_intelligence.knowledge.store import KnowledgeStore
        store = KnowledgeStore(".")
        df = pl.DataFrame({"x": pl.Series("x", ['{"a":1}'], dtype=pl.String)})
        # Should not raise
        path = store.write_asset("evidence", df)
        Path(path).unlink()


class TestProblemSignalToDataframe:
    """ProblemSignal has no dict fields, but verify it exports cleanly."""

    def test_problem_signal_export(self, tmp_path: Path) -> None:
        sig = ProblemSignal(
            signal_key="test:value:",
            category="shipping",
            entity="Amazon",
            entity_type=EntityType.COMPANY,
            signal_text="late delivery",
            document_count=10,
            avg_rating=2.5,
            evidence_ids=["ev1", "ev2"],
            observation_count=15,
            confidence=0.85,
            pipeline_version="1.5.0",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        df = ProblemSignal.to_dataframe([sig])
        path = tmp_path / "problem_signals.parquet"
        df.write_parquet(path)
        reloaded = pl.read_parquet(path)
        assert len(reloaded) == 1
        assert reloaded["signal_key"][0] == "test:value:"


class TestEndToEndWriteAssets:
    """Full cycle: Evidence.to_dataframe() → store.write_asset() → read back."""

    def test_evidence_roundtrip(self) -> None:
        from pain_intelligence.knowledge.store import KnowledgeStore
        from pathlib import Path
        import tempfile
        import shutil

        tmp = Path(tempfile.mkdtemp())
        try:
            store = KnowledgeStore(tmp)
            ev = Evidence(
                evidence_id="ev1",
                signal_key="test:value:",
                platform_distribution={"amazon": 5},
                country_distribution={},
            )
            df = Evidence.to_dataframe([ev])
            path = store.write_asset("evidence", df)
            assert path.exists()

            reloaded = store.read_asset("evidence")
            assert reloaded["platform_distribution"].dtype == pl.String
            assert json.loads(reloaded["platform_distribution"][0]) == {"amazon": 5}
            assert json.loads(reloaded["country_distribution"][0]) == {}
        finally:
            shutil.rmtree(tmp)

    def test_multiple_evidence_write(self) -> None:
        from pain_intelligence.knowledge.store import KnowledgeStore
        from pathlib import Path
        import tempfile
        import shutil

        tmp = Path(tempfile.mkdtemp())
        try:
            store = KnowledgeStore(tmp)
            evs = [
                Evidence(
                    evidence_id=f"ev{i}",
                    signal_key=f"test:{i}:",
                    platform_distribution={"amazon": i} if i % 2 == 0 else {},
                    country_distribution={} if i % 2 == 0 else {"us": i},
                )
                for i in range(10)
            ]
            df = Evidence.to_dataframe(evs)
            path = store.write_asset("evidence", df)
            assert path.exists()

            reloaded = store.read_asset("evidence")
            assert len(reloaded) == 10
            assert reloaded["platform_distribution"].dtype == pl.String
            assert reloaded["country_distribution"].dtype == pl.String
        finally:
            shutil.rmtree(tmp)
